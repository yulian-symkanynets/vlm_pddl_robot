"""Adaptive multi-strategy planner (architecture O in the guide).

A fixed, rule-based order of stages. Each stage either answers or passes on, and
the trace records which stage answered and what every stage cost.

    0. instruction -> goal        only if a natural-language instruction came with the scene
                                  (the one place a language model is needed)
    1. uncertainty gate           unidentified objects: act on what all hypotheses agree on,
                                  or request another view
    2. experience                 exact structural case lookup
    3. symbolic planner           graph planner, A* fallback inside
    4. second symbolic planner    optional, e.g. PDDL for goals the graph planner cannot do
    5. model fallback             optional LLM with validator-in-the-loop; reached only if
                                  every symbolic stage failed
    final                         whatever comes out is validated by run()

A learned selector is not used on purpose: stages 2 and 3 cost well under a
millisecond, so there is nothing for a selector to save, and a fixed order is
reproducible. Learned selection would start to pay only if several expensive
stages competed.
"""
from __future__ import annotations

import time

from .active_perception import decide
from .base import Proposal, Strategy
from .symbolic import CaseBasedStrategy, GraphStrategy
from semplan.scene import Scene


class AdaptivePlanner:
    name = "adaptive"

    def __init__(self, llm_goal: Strategy | None = None, second: Strategy | None = None,
                 llm_fallback: Strategy | None = None,
                 color_probs: dict[str, dict[str, float]] | None = None):
        self.cases = CaseBasedStrategy(GraphStrategy())      # stages 2 + 3
        self.llm_goal = llm_goal                             # an LLMPlusPStrategy with instruction
        self.second = second
        self.llm_fallback = llm_fallback
        self.color_probs = color_probs or {}
        self.trace: list[dict] = []
        self.answered: dict[str, int] = {}

    def _note(self, stage: str, t0: float, result: str) -> None:
        self.trace.append({"stage": stage, "ms": round((time.perf_counter() - t0) * 1e3, 4),
                           "result": result})

    def propose(self, scene: Scene) -> Proposal:
        self.trace = []
        calls = 0
        # 0. instruction -> goal
        if self.llm_goal is not None and getattr(self.llm_goal, "instruction", None):
            t0 = time.perf_counter()
            before = self.llm_goal.llm.calls
            try:
                goals = self.llm_goal.translate(scene, self.llm_goal.instruction)
            except Exception as e:
                self._note("instruction->goal", t0, f"rejected: {e}")
                return self._done("instruction->goal", Proposal(None, f"goal rejected: {e}", "llm",
                                                                self.llm_goal.llm.calls - before))
            calls += self.llm_goal.llm.calls - before
            scene = scene.copy()
            scene.explicit_goals = goals
            self._note("instruction->goal", t0, f"{len(goals)} goal(s)")
        # 1. uncertainty gate
        unknown = [o for o in scene.objects if not scene.trusted(o)]
        if unknown and scene.explicit_goals is None:
            t0 = time.perf_counter()
            d = decide(scene, self.color_probs)
            if d["action"] == "REQUEST_NEW_VIEW":
                self._note("uncertainty", t0, f"request view of {d['target']}")
                return self._done("uncertainty", Proposal(
                    None, d["reason"], "uncertainty", calls,
                    [{"action": "REQUEST_NEW_VIEW", "target": d["target"]}], scene))
            self._note("uncertainty", t0, f"{len(d['moves'])} agreed move(s)")
        # 2 + 3. experience, then symbolic planner
        t0 = time.perf_counter()
        p = self.cases.propose(scene)
        self._note("case+graph", t0, p.source if p.moves is not None else f"failed: {p.reason}")
        stage = "case" if p.source == "case" else "graph"
        # 4. second symbolic planner
        if p.moves is None and self.second is not None:
            t0 = time.perf_counter()
            p = self.second.propose(scene)
            stage = self.second.name
            self._note(stage, t0, "ok" if p.moves is not None else f"failed: {p.reason}")
        # 5. model fallback
        if p.moves is None and self.llm_fallback is not None:
            t0 = time.perf_counter()
            p = self.llm_fallback.propose(scene)
            calls += p.llm_calls
            stage = self.llm_fallback.name
            self._note(stage, t0, "ok" if p.moves is not None else f"failed: {p.reason}")
        p.llm_calls = calls
        p.scene = scene
        if p.moves is not None:
            p.info_requests = [{"action": "REQUEST_NEW_VIEW", "target": o} for o in sorted(unknown)]
        return self._done(stage if p.moves is not None else "none", p)

    def _done(self, stage: str, p: Proposal) -> Proposal:
        self.answered[stage] = self.answered.get(stage, 0) + 1
        p.source = f"{stage} | " + " > ".join(f"{t['stage']}({t['result']})" for t in self.trace)
        return p
