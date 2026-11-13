"""Contingent (conditional) plans: one answer that already contains the branches.

With an unidentified object the ordinary flow is: plan what is safe, ask for a
view, wait for the new scene, plan again. A contingent plan does the second
planning in advance. The server returns a small tree:

    moves that are right whatever the colour turns out to be
    OBSERVE unknown_1
      if blue  -> moves ...
      if green -> moves ...

The laptop executes the first moves, looks, and continues down the matching
branch without another round trip to the server. The branches are computed
with the ordinary planner, once per hypothesis; every root-to-leaf path is a
complete valid plan for its hypothesis (checked in the tests).

The tree grows with (colours ^ unknown objects), so this is for one or two
uncertain objects, which is the situation on a small table.
"""
from __future__ import annotations

from dataclasses import replace

from .active_perception import decide
from .base import Proposal, Strategy
from .symbolic import GraphStrategy
from semplan.scene import Move, Scene, apply


def build_tree(scene: Scene, color_probs: dict[str, dict[str, float]],
               planner: Strategy | None = None, depth: int = 0) -> dict:
    """-> {"moves": [...], "observe": id | None, "branches": {colour: subtree}, "complete": bool}"""
    planner = planner or GraphStrategy()
    unknown = [o for o in sorted(scene.objects) if not scene.trusted(o)]
    if not unknown or depth > 4:
        p = planner.propose(scene)
        return {"moves": p.moves or [], "observe": None, "branches": {}, "complete": p.moves is not None}
    d = decide(scene, color_probs, planner)
    prefix: list[Move] = d["moves"] if d["action"] == "EXECUTE" else []
    after = scene.copy()
    for m in prefix:
        apply(after, m)
    if d["action"] == "EXECUTE" and d["complete"]:
        return {"moves": prefix, "observe": None, "branches": {}, "complete": True}
    target = d.get("target") or unknown[0]
    zone_colors = sorted(l.color for l in scene.locs.values() if l.kind == "zone")
    colours = sorted(color_probs.get(target) or zone_colors,
                     key=lambda c: -(color_probs.get(target, {}).get(c, 0.0)))
    branches = {}
    for c in colours:
        s = after.copy()
        s.objects = dict(s.objects)
        s.objects[target] = replace(s.objects[target], color=c, confidence=1.0)
        branches[c] = build_tree(s, color_probs, planner, depth + 1)
    return {"moves": prefix, "observe": target, "branches": branches,
            "complete": all(b["complete"] for b in branches.values())}


def follow(tree: dict, truth: dict[str, str]) -> list[Move]:
    """The move sequence the robot ends up executing if the hidden colours are `truth`."""
    moves = list(tree["moves"])
    while tree["observe"] is not None:
        tree = tree["branches"][truth[tree["observe"]]]
        moves += tree["moves"]
    return moves


def render(tree: dict, indent: str = "") -> str:
    lines = [f"{indent}{m.obj} -> {m.dst}" for m in tree["moves"]] or [f"{indent}(nothing to do yet)"]
    if tree["observe"] is not None:
        lines.append(f"{indent}OBSERVE {tree['observe']}")
        for c, sub in tree["branches"].items():
            lines.append(f"{indent}  if {c}:")
            lines.append(render(sub, indent + "    "))
    return "\n".join(lines)


def count_leaves(tree: dict) -> int:
    return 1 if tree["observe"] is None else sum(count_leaves(b) for b in tree["branches"].values())


class ContingentStrategy:
    """Returns the unconditional first moves plus the request to observe; the full
    conditional plan is in `.tree` (send it along so the laptop can continue alone)."""

    def __init__(self, color_probs: dict[str, dict[str, float]] | None = None,
                 planner: Strategy | None = None):
        self.name = "contingent"
        self.color_probs, self.planner = color_probs or {}, planner or GraphStrategy()
        self.tree: dict | None = None

    def propose(self, scene: Scene) -> Proposal:
        self.tree = build_tree(scene, self.color_probs, self.planner)
        if self.tree["observe"] is None:
            return Proposal(self.tree["moves"] if self.tree["complete"] else None, source="contingent")
        return Proposal(self.tree["moves"], source=f"contingent tree, {count_leaves(self.tree)} leaves",
                        info_requests=[{"action": "REQUEST_NEW_VIEW", "target": self.tree["observe"]}],
                        complete=False)
