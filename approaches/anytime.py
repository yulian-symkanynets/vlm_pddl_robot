"""Anytime planning: a valid plan immediately, a better one if there is time.

  1. The constructive planner gives a valid plan at once (tens of microseconds).
  2. Redundant moves are removed: a move is dropped if the plan is still valid
     without it (cheap clean-up of detours).
  3. With the remaining time budget A* searches for a shorter plan; if the
     budget ends first, the best plan so far is returned.

The answer is never late and never invalid; only its length depends on the
budget. Useful when the robot is still busy with the previous step: the server
can spend that idle time on a better plan for the next one.
"""
from __future__ import annotations

import time

from .base import Proposal
from semplan.planner import plan_astar, plan_greedy
from semplan.scene import Move, Scene, validate


def eliminate_redundant(scene: Scene, moves: list[Move], deadline: float | None = None) -> list[Move]:
    """Drop moves (or parked-then-moved pairs) whose removal keeps the plan valid.
    Stops at `deadline` (a time.perf_counter() value) and returns what it has."""
    improved = True
    while improved:
        improved = False
        if deadline is not None and time.perf_counter() > deadline:
            break
        for i in range(len(moves)):
            trial = moves[:i] + moves[i + 1:]
            if validate(scene, trial).ok:
                moves, improved = trial, True
                break
            for j in range(i + 1, len(moves)):              # merge "o -> X ... o -> Y" into "o -> Y"
                if moves[j].obj == moves[i].obj:
                    trial = moves[:i] + moves[i + 1:j] + [moves[j]] + moves[j + 1:]
                    trial2 = moves[:i] + [moves[j]] + moves[i + 1:j] + moves[j + 1:]
                    for t in (trial, trial2):
                        if len(t) < len(moves) and validate(scene, t).ok:
                            moves, improved = t, True
                            break
                    break
            if improved:
                break
    return moves


class AnytimeStrategy:
    def __init__(self, budget_ms: float = 5.0):
        self.name = f"anytime({budget_ms:g} ms)"
        self.budget_ms = budget_ms
        self.stages: list[tuple[str, float, int]] = []       # (stage, ms since start, plan length)

    def propose(self, scene: Scene) -> Proposal:
        t0 = time.perf_counter()
        elapsed = lambda: (time.perf_counter() - t0) * 1e3
        self.stages = []
        first = plan_greedy(scene)
        best = first.moves if first.status == "ok" else None
        if best is not None:
            self.stages.append(("constructive", elapsed(), len(best)))
            if self.budget_ms > 0:
                cleaned = eliminate_redundant(scene, best, t0 + self.budget_ms / 2e3)
                if len(cleaned) < len(best):
                    best = cleaned
                    self.stages.append(("redundancy removal", elapsed(), len(best)))
        left = self.budget_ms - elapsed()
        if left > 0.05 or best is None:
            r = plan_astar(scene, 10 ** 9, max(left, 50.0 if best is None else left))
            if r.status == "ok" and (best is None or len(r.moves) < len(best)):
                best = r.moves
                self.stages.append(("A* improvement", elapsed(), len(best)))
        if best is None:
            return Proposal(None, "no plan within the budget", "anytime")
        return Proposal(best, source=" > ".join(s for s, _, _ in self.stages))
