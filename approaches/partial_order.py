"""Partial-order plans: a plan as a dependency graph instead of a fixed sequence.

A total-order plan says "do step 1, then 2, then 3". Most of that order is
accidental: delivering the green cube and delivering the blue cylinder have
nothing to do with each other. A partial-order plan keeps only the orderings
that matter (step B needs step A to have happened) and leaves the rest open.

What that buys:
  * the executor may pick any step whose predecessors are done, for example the
    one closest to the gripper, or skip an object that is momentarily unreachable;
  * the server can tell which steps a change can affect (the changed objects'
    steps and everything after them) and which are untouched.

How the graph is built here (deordering of a valid total-order plan): each step
uses resources (the object it moves, what that object stood on, where it goes).
Two steps are ordered only if they share a resource; then the order of the
original plan is kept. Every ordering of the steps that respects the graph is a
valid plan (checked in the tests by sampling orderings through the validator).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .base import Proposal, Strategy
from .symbolic import GraphStrategy
from semplan.scene import HELD, Move, Scene, apply


@dataclass
class PartialOrderPlan:
    steps: list[Move]
    before: dict[int, set[int]] = field(default_factory=dict)   # step -> steps that must come first

    def ready(self, done: set[int]) -> list[int]:
        """Steps that may be executed now."""
        return [i for i in range(len(self.steps)) if i not in done and self.before[i] <= done]

    def edges(self) -> list[tuple[int, int]]:
        """Direct dependencies only (transitive reduction), as (earlier, later)."""
        out = []
        for j, pre in self.before.items():
            for i in pre:
                if not any(i in self.before[k] for k in pre if k != i):
                    out.append((i, j))
        return sorted(out)

    def flexibility(self) -> float:
        """Share of step pairs left unordered: 0 = a fixed sequence, 1 = any order."""
        n = len(self.steps)
        if n < 2:
            return 0.0
        return 1 - sum(len(p) for p in self.before.values()) / (n * (n - 1) / 2)

    def affected_by(self, changed_objects: set[str]) -> set[int]:
        """Steps that touch a changed object, plus every step that depends on them."""
        hit = {i for i, m in enumerate(self.steps) if m.obj in changed_objects or m.dst in changed_objects}
        grew = True
        while grew:
            grew = False
            for j, pre in self.before.items():
                if j not in hit and pre & hit:
                    hit.add(j)
                    grew = True
        return hit

    def linearize(self, cost=None) -> list[Move]:
        """One valid ordering. With `cost(previous_step, next_step)` the cheapest ready
        step is taken each time (greedy), e.g. to shorten gripper travel."""
        done: set[int] = set()
        order, last = [], None
        while len(done) < len(self.steps):
            ready = self.ready(done)
            i = min(ready, key=(lambda k: (cost(last, k), k)) if cost else (lambda k: k))
            order.append(i)
            done.add(i)
            last = i
        return [self.steps[i] for i in order]


def deorder(scene: Scene, moves: list[Move]) -> PartialOrderPlan:
    """Remove the orderings of a valid plan that no resource conflict requires."""
    s = scene.copy()
    floor_puts: dict[str, int] = {}
    for m in moves:
        if m.dst in s.locs:
            floor_puts[m.dst] = floor_puts.get(m.dst, 0) + 1
    # A location is "roomy" if all placements of the plan fit without anything leaving first;
    # then steps that only share that location do not depend on each other.
    roomy = {l for l, k in floor_puts.items() if scene.floor_free(l) >= k}
    uses: list[set[str]] = []
    for m in moves:
        src = s.on[m.obj]
        res = {m.obj}
        for x in (src, m.dst):
            if x in s.objects:
                res.add(x)
            elif x != HELD and s.locs[x].kind != "table" and x not in roomy:
                res.add(x)
        uses.append(res)
        apply(s, m)
    before: dict[int, set[int]] = {i: set() for i in range(len(moves))}
    for j in range(len(moves)):
        for i in range(j):
            if uses[i] & uses[j]:
                before[j] |= before[i] | {i}                   # keep it transitively closed
    return PartialOrderPlan(list(moves), before)


def travel_cost(scene: Scene, plan: PartialOrderPlan, positions: dict[str, tuple[float, float]]):
    """cost(prev, next) = distance from where step `prev` put its object to where step
    `next` picks its object up. `positions` maps object and location ids to (x, y)."""
    s = scene.copy()
    pick_at, place_at = {}, {}
    for i, m in enumerate(plan.steps):                          # positions along the original plan
        src = s.on[m.obj]
        pick_at[i] = positions.get(m.obj) or positions.get(src) or (0.0, 0.0)
        place_at[i] = positions.get(m.dst) or (0.0, 0.0)
        positions = dict(positions, **{m.obj: place_at[i]})
        apply(s, m)

    def cost(prev, nxt):
        a = place_at[prev] if prev is not None else positions.get("__gripper__", (0.0, 0.0))
        b = pick_at[nxt]
        return math.hypot(a[0] - b[0], a[1] - b[1])
    return cost


class PartialOrderStrategy:
    """Plans with any strategy, then returns the plan as a dependency graph.

    propose() still returns a valid sequence (so it fits the common interface);
    the graph is in `.last` for an executor that wants the freedom.
    """

    def __init__(self, planner: Strategy | None = None,
                 positions: dict[str, tuple[float, float]] | None = None):
        self.name = "partial_order"
        self.planner = planner or GraphStrategy()
        self.positions = positions
        self.last: PartialOrderPlan | None = None

    def propose(self, scene: Scene) -> Proposal:
        p = self.planner.propose(scene)
        if p.moves is None:
            return p
        self.last = deorder(scene, p.moves)
        cost = travel_cost(scene, self.last, self.positions) if self.positions else None
        return Proposal(self.last.linearize(cost), source=f"{p.source} -> partial order")
