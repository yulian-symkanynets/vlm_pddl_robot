"""Planning with an unidentified object: act, or look first?

A light alternative to full belief-space (POMDP) planning, suitable when only a
few facts are uncertain:

  1. Enumerate the hypotheses (possible colours of each unidentified object,
     with the probabilities the laptop reports in "color_probs").
  2. Plan for every hypothesis with the ordinary planner (determinisation).
  3. The moves that all plans agree on, from the start, are safe to execute now.
  4. If the plans disagree on the very first move, the missing information matters
     right now: request a view of the object whose identity splits the plans most.

So the robot gathers information only when the information changes what it would do.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import replace

from .base import Proposal, Strategy
from .symbolic import GraphStrategy
from semplan.scene import Move, Scene

MAX_HYPOTHESES = 64


def hypotheses(scene: Scene, color_probs: dict[str, dict[str, float]]):
    """Yield (probability, scene with every unidentified object given a colour)."""
    unknown = [o for o in sorted(scene.objects) if not scene.trusted(o)]
    zone_colors = sorted(l.color for l in scene.locs.values() if l.kind == "zone")
    options = []
    for o in unknown:
        probs = color_probs.get(o) or {c: 1 / len(zone_colors) for c in zone_colors}
        total = sum(probs.values())
        options.append([(c, p / total) for c, p in sorted(probs.items(), key=lambda kv: -kv[1])])
    combos = list(itertools.product(*options))[:MAX_HYPOTHESES]
    for combo in combos:
        s = scene.copy()
        s.objects = dict(s.objects)
        p = 1.0
        for o, (c, pc) in zip(unknown, combo):
            s.objects[o] = replace(s.objects[o], color=c, confidence=1.0)
            p *= pc
        yield p, s, dict(zip(unknown, (c for c, _ in combo)))


def decide(scene: Scene, color_probs: dict[str, dict[str, float]],
           planner: Strategy | None = None) -> dict:
    """-> {"action": "EXECUTE", "moves": [...]} or {"action": "REQUEST_NEW_VIEW", "target": id}"""
    planner = planner or GraphStrategy()
    plans = []
    for p, s, assignment in hypotheses(scene, color_probs):
        prop = planner.propose(s)
        plans.append((p, prop.moves or [], assignment))
    if len(plans) <= 1:
        return {"action": "EXECUTE", "moves": plans[0][1] if plans else [], "hypotheses": len(plans),
                "complete": True}
    prefix: list[Move] = []
    for step in zip(*(pl for _, pl, _ in plans)):
        if len(set(step)) != 1:
            break
        prefix.append(step[0])
    complete = all(len(pl) == len(prefix) for _, pl, _ in plans)
    if prefix:
        return {"action": "EXECUTE", "moves": prefix, "hypotheses": len(plans), "complete": complete}
    # The first move already depends on the unknown: which object's colour splits it most?
    def entropy(o):
        dist: dict = {}
        for p, pl, a in plans:
            key = (a[o], pl[0] if pl else None)
            dist[key] = dist.get(key, 0.0) + p
        by_colour: dict = {}
        for (c, first), p in dist.items():
            by_colour.setdefault(first, 0.0)
            by_colour[first] += p
        return -sum(p * math.log(p) for p in by_colour.values() if p > 0)
    unknown = list(plans[0][2])
    target = max(unknown, key=entropy)
    return {"action": "REQUEST_NEW_VIEW", "target": target, "hypotheses": len(plans),
            "reason": "the first move depends on the colour of this object"}


class ActivePerceptionStrategy:
    """Wraps a planner: returns the agreed prefix, or an information request."""

    def __init__(self, color_probs: dict[str, dict[str, float]] | None = None,
                 planner: Strategy | None = None):
        self.name = "active_perception"
        self.color_probs = color_probs or {}
        self.planner = planner or GraphStrategy()

    def propose(self, scene: Scene) -> Proposal:
        d = decide(scene, self.color_probs, self.planner)
        if d["action"] == "REQUEST_NEW_VIEW":
            return Proposal(None, d["reason"], "active_perception",
                            info_requests=[{"action": "REQUEST_NEW_VIEW", "target": d["target"]}])
        pending = [{"action": "REQUEST_NEW_VIEW", "target": o, "when": "after these moves"}
                   for o in sorted(scene.objects) if not scene.trusted(o)] if not d["complete"] else []
        return Proposal(d["moves"], source=f"agreed prefix of {d['hypotheses']} hypotheses",
                        info_requests=pending, complete=d["complete"])
