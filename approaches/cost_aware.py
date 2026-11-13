"""Failure-aware planning: prefer the plan that is most likely to work first time.

The symbolic model treats every move as equally good. The robot knows better:
some grasps and placements fail far more often than others, and the server
collects exactly these statistics from step feedback (semplan.cases.ActionStats).

Here every move gets a cost equal to the expected number of attempts,
1 / (p_pick * p_place), and a uniform-cost search (A* with an admissible
heuristic) returns the plan with the lowest expected number of attempts
instead of the fewest moves. With all probabilities equal to 1 it reduces to
the ordinary optimal planner.
"""
from __future__ import annotations

import heapq

from .base import Proposal, Strategy
from .symbolic import GraphStrategy
from semplan.cases import ActionStats
from semplan.scene import HELD, Move, Scene, apply, legal_dests


def move_success(scene: Scene, m: Move, stats: ActionStats) -> float:
    """Estimated probability that PICK and PLACE of this move both succeed."""
    src = scene.on[m.obj]
    p = 1.0
    if src != HELD:
        p *= stats.p_success(stats.context(scene, "PICK", m.obj, src))
    return p * stats.p_success(stats.context(scene, "PLACE", m.obj, m.dst))


def expected_attempts(scene: Scene, moves: list[Move], stats: ActionStats) -> float:
    """Expected number of move attempts for a plan, if failed moves are simply retried."""
    s, total = scene.copy(), 0.0
    for m in moves:
        total += 1.0 / max(move_success(s, m, stats), 1e-6)
        apply(s, m)
    return total


def plan_min_expected_attempts(scene: Scene, stats: ActionStats, max_expansions: int = 30_000):
    """A* on expected attempts. Each object that still has to move costs at least 1,
    so the count of such objects is an admissible heuristic."""
    goals = scene.goals()

    def h(s: Scene) -> float:
        return float(len(s.objects) - len(s.settled(goals)) if s.held() is None
                     else len(s.objects) - len(s.settled(goals)))

    start = scene.copy()
    counter = 0
    frontier = [(h(start), 0.0, counter, start, [])]
    best = {tuple(sorted(start.on.items())): 0.0}
    expansions = 0
    while frontier:
        _, g, _, s, path = heapq.heappop(frontier)
        if s.goal_reached():
            return path, expansions
        if g > best.get(tuple(sorted(s.on.items())), float("inf")):
            continue
        expansions += 1
        if expansions > max_expansions:
            return None, expansions
        held = s.held()
        movers = [held] if held else sorted(o for o in s.clear() if s.movable(o))
        for o in movers:
            for d in legal_dests(s, o):
                if d == s.on[o]:
                    continue
                m = Move(o, d)
                g2 = g + 1.0 / max(move_success(s, m, stats), 1e-6)
                s2 = s.copy()
                apply(s2, m)
                key = tuple(sorted(s2.on.items()))
                if g2 >= best.get(key, float("inf")):
                    continue
                best[key] = g2
                counter += 1
                heapq.heappush(frontier, (g2 + h(s2), g2, counter, s2, path + [m]))
    return None, expansions


class CostAwareStrategy:
    """Minimises expected attempts using learned success statistics.

    Falls back to the ordinary planner if the search budget runs out (larger scenes).
    """

    def __init__(self, stats: ActionStats, fallback: Strategy | None = None,
                 max_expansions: int = 30_000):
        self.name = "cost_aware"
        self.stats, self.fallback = stats, fallback or GraphStrategy()
        self.max_expansions = max_expansions

    def propose(self, scene: Scene) -> Proposal:
        moves, n = plan_min_expected_attempts(scene, self.stats, self.max_expansions)
        if moves is None:
            p = self.fallback.propose(scene)
            p.source = f"budget exhausted -> {p.source}"
            return p
        return Proposal(moves, source=f"expected-attempts A* ({n} expansions)")
