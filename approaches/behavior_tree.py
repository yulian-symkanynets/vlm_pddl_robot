"""Behaviour tree with symbolic recovery (architecture N).

A behaviour tree is ticked over and over. Each tick walks the tree from the root
and returns SUCCESS, FAILURE or RUNNING. Two composite nodes are enough here:

  Sequence   ticks children left to right, stops at the first that is not SUCCESS
  Fallback   ticks children left to right, stops at the first that is not FAILURE

The tree below re-checks its conditions on every tick, so recovery needs no
special code path: if the plan stops matching the observed scene, the condition
"plan is valid" fails and the tree falls through to "replan" by itself.

    Fallback
    |- Condition: goal reached
    `- Sequence
       |- Action:   observe
       |- Fallback
       |  |- Condition: nothing important is unidentified
       |  `- Action:    request another view
       |- Fallback
       |  |- Condition: remaining plan is valid on the observed scene
       |  `- Action:    replan (symbolic planner)
       `- Action:   execute the next step

The planner supplies the knowledge, the tree supplies the reactivity. For a
production tree use py_trees (Python) or BehaviorTree.CPP; this file only shows
the mechanism in ~100 lines.
"""
from __future__ import annotations

from typing import Callable

from .base import Strategy
from .symbolic import GraphStrategy
from semplan.scene import parse_scene, to_contract, validate

SUCCESS, FAILURE, RUNNING = "SUCCESS", "FAILURE", "RUNNING"


class Node:
    def __init__(self, name: str, children: list["Node"] | None = None):
        self.name, self.children = name, children or []

    def tick(self, bb: dict) -> str:
        raise NotImplementedError


class Sequence(Node):
    def tick(self, bb):
        for c in self.children:
            status = c.tick(bb)
            if status != SUCCESS:
                return status
        return SUCCESS


class Fallback(Node):
    def tick(self, bb):
        for c in self.children:
            status = c.tick(bb)
            if status != FAILURE:
                return status
        return FAILURE


class Condition(Node):
    def __init__(self, name: str, test: Callable[[dict], bool]):
        super().__init__(name)
        self.test = test

    def tick(self, bb):
        return SUCCESS if self.test(bb) else FAILURE


class Action(Node):
    def __init__(self, name: str, act: Callable[[dict], str]):
        super().__init__(name)
        self.act = act

    def tick(self, bb):
        status = self.act(bb)
        bb["trace"].append(f"{self.name}: {status}")
        return status


# ---------------------------------------------------------------- leaves

def _observe(bb):
    bb["scene"] = parse_scene(bb["robot"].observe())
    return SUCCESS


def _plan_valid(bb):
    return bool(bb["plan"]) and validate(bb["scene"], bb["plan"]).ok


def _replan(bb):
    p = bb["planner"].propose(bb["scene"])
    bb["replans"] += 1
    if p.moves is None:
        return FAILURE
    bb["plan"] = p.moves
    return SUCCESS if p.moves else FAILURE


def _identified(bb):
    return not any(why in ("unidentified", "low_identity_confidence", "occluded")
                   for why in bb["scene"].deferred().values())


def _request_view(bb):
    asked = False
    for o, why in sorted(bb["scene"].deferred().items()):
        if bb["robot"].fulfil({"action": "REQUEST_NEW_VIEW", "target": o}):
            asked = True
    bb["views"] += asked
    return RUNNING if asked else SUCCESS       # RUNNING: look again on the next tick


def _execute_next(bb):
    move = bb["plan"][0]
    for step in to_contract(bb["scene"], [move]):
        bb["steps"] += 1
        if bb["robot"].execute(step)["status"] != "success":
            return RUNNING                     # failed step: next tick observes and decides again
    bb["plan"] = bb["plan"][1:]
    return RUNNING


def build_tree() -> Node:
    return Fallback("root", [
        Condition("goal reached", lambda bb: bb["scene"] is not None and bb["scene"].goal_reached()
                  and not bb["scene"].deferred() and bb["robot"].truth.held() is None
                  and bb.pop("fresh", False)),
        Sequence("work", [
            Action("observe", _observe),
            Fallback("perception", [Condition("identified", _identified),
                                    Action("request view", _request_view)]),
            Fallback("planning", [Condition("plan valid", _plan_valid), Action("replan", _replan)]),
            Action("execute next", _execute_next),
        ]),
    ])


def run_tree(robot, planner: Strategy | None = None, max_ticks: int = 300,
             disturb: Callable[[int], None] | None = None) -> dict:
    """Tick the tree until the goal is reached. `disturb(tick)` may change the world."""
    bb = {"robot": robot, "planner": planner or GraphStrategy(), "scene": None, "plan": [],
          "replans": 0, "steps": 0, "views": 0, "trace": []}
    root = build_tree()
    for tick in range(max_ticks):
        if disturb:
            disturb(tick)
        if bb["scene"] is not None:            # the goal check must use a fresh observation
            bb["scene"] = parse_scene(robot.observe())
            bb["fresh"] = True
        status = root.tick(bb)
        if status == SUCCESS:
            return {"goal": True, "ticks": tick + 1, **{k: bb[k] for k in ("replans", "steps", "views", "trace")}}
        if status == FAILURE:
            break
    return {"goal": False, "ticks": max_ticks, **{k: bb[k] for k in ("replans", "steps", "views", "trace")}}
