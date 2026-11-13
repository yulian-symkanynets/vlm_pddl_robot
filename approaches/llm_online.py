"""Strategies that call a language model while handling a request.

  LLMDirectStrategy   the model writes the plan                     (architecture B)
  LLMModuloStrategy   the model writes, the validator criticises,
                      the model corrects, up to k rounds            (architecture K without retrieval)
  LLMPlusPStrategy    the model only translates an instruction into
                      a goal; a symbolic planner makes the plan     (architecture D, LLM+P)

All three are judged by the same validator as every other strategy.
"""
from __future__ import annotations

import json

from .base import Proposal, Strategy
from .llm import LLM, extract_json, extract_moves
from .symbolic import GraphStrategy
from semplan.scene import Move, Scene, SceneError, dump_scene, parse_goal, validate

PLAN_PROMPT = """You plan for a robot arm that sorts objects into colour zones.
One action moves one object:  MOVE <object> <destination>
Rules:
- Only an object with nothing on top of it can be moved.
- A destination is a zone or free slot with a free ground pose, or an object with
  a flat top and nothing on it. Nothing can be put on a prism or on the open table.
- "capacity" is the number of ground poses of a zone. A free slot holds one object.
- Rules of the task (if any) hold for every single action, not only at the end.
Task: {goal}
Task JSON: {task_json}
Scene:
{scene}
{feedback}Answer with one MOVE line per action, in order, and nothing else."""

GOAL_PROMPT = """Translate the instruction into goals for a sorting robot.
Use only these object ids: {objects}
Use only these zone ids: {zones}
Instruction: {instruction}
Answer with JSON only: {{"goal": {{"<object id>": "<zone id>", ...}}}}"""


def describe_goal(scene: Scene) -> str:
    if scene.task is not None:
        return scene.task.describe()
    if scene.explicit_goals is not None:
        return "; ".join(f"{o} must end in {z}" for o, z in sorted(scene.explicit_goals.items()))
    return "every object must end in the zone of its own colour"


def task_json(scene: Scene) -> str:
    if scene.task is not None:
        return json.dumps(scene.task.raw, separators=(",", ":"))
    if scene.explicit_goals is not None:
        return json.dumps({"goal": [{"object": o, "in": z} for o, z in sorted(scene.explicit_goals.items())]},
                          separators=(",", ":"))
    return '{"goal":[{"sort_by_color":true}]}'


def scene_text(scene: Scene) -> str:
    d = dump_scene(scene)
    d.pop("goal", None)
    d.pop("task", None)
    return json.dumps(d, separators=(",", ":"))


class LLMDirectStrategy:
    """One model call, the answer is the plan. No guarantee of any kind."""

    def __init__(self, llm: LLM):
        self.name, self.llm = "llm_direct", llm

    def propose(self, scene: Scene, feedback: str = "") -> Proposal:
        before = self.llm.calls
        text = self.llm.complete(PLAN_PROMPT.format(goal=describe_goal(scene), task_json=task_json(scene),
                                                    scene=scene_text(scene), feedback=feedback))
        moves = [Move(o, d) for o, d in extract_moves(text)]
        return Proposal(moves, source="llm", llm_calls=self.llm.calls - before)


class LLMModuloStrategy:
    """Generate-test-critique loop (LLM-Modulo; the same loop as ViLaIn's corrective
    re-prompting and Inner Monologue's feedback, with a sound symbolic critic).

    The validator's error message is given back to the model. The loop ends with
    the first plan that validates, or with no plan after `max_rounds`.
    """

    def __init__(self, llm: LLM, max_rounds: int = 3):
        self.name, self.llm, self.max_rounds = "llm_modulo", llm, max_rounds
        self._direct = LLMDirectStrategy(llm)

    def propose(self, scene: Scene) -> Proposal:
        before, feedback, reason = self.llm.calls, "", ""
        for round_no in range(1, self.max_rounds + 1):
            p = self._direct.propose(scene, feedback)
            v = validate(scene, p.moves)
            if v.ok:
                return Proposal(p.moves, source=f"llm round {round_no}",
                                llm_calls=self.llm.calls - before)
            reason = v.reason
            feedback = ("Your previous plan was rejected by the validator:\n"
                        + "\n".join(str(m) for m in p.moves) + f"\nError: {v.reason}\n")
        return Proposal(None, f"no valid plan after {self.max_rounds} rounds: {reason}", "llm",
                        self.llm.calls - before)


class LLMPlusPStrategy:
    """LLM+P: the model translates, the planner plans.

    Here the scene already arrives in symbolic form, so the only thing left to
    translate is a natural-language instruction into a goal. The goal is checked
    against the scene (unknown ids are rejected) before any planning happens.
    With no instruction the model is not called at all.
    """

    def __init__(self, llm: LLM, planner: Strategy | None = None, instruction: str | None = None):
        self.name, self.llm = "llm+p", llm
        self.planner = planner or GraphStrategy()
        self.instruction = instruction

    def translate(self, scene: Scene, instruction: str) -> dict[str, str]:
        zones = sorted(l.id for l in scene.locs.values() if l.kind == "zone")
        text = self.llm.complete(GOAL_PROMPT.format(objects=sorted(scene.objects), zones=zones,
                                                    instruction=instruction))
        js = extract_json(text)
        if not isinstance(js, dict) or "goal" not in js:
            raise SceneError("model did not return a goal")
        return parse_goal(js["goal"], scene.objects, scene.locs)      # raises on unknown ids

    def propose(self, scene: Scene) -> Proposal:
        before = self.llm.calls
        if self.instruction:
            try:
                goals = self.translate(scene, self.instruction)
            except (SceneError, KeyError, TypeError, AttributeError) as e:
                return Proposal(None, f"goal translation rejected: {e}", "llm",
                                self.llm.calls - before)
            if not goals:
                return Proposal(None, "instruction did not name any known object", "llm",
                                self.llm.calls - before)
            scene = scene.copy()
            scene.explicit_goals = goals
        p = self.planner.propose(scene)
        p.scene = scene
        p.llm_calls = self.llm.calls - before
        p.source = f"goal:{'llm' if self.instruction else 'default'} plan:{p.source}"
        return p
