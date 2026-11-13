"""General task planner: any goals and constraints of semplan.task.Task.

A* over scenes. Successors are the moves the action model allows (physical
preconditions, task constraints, collision blockers), so every plan it returns
passes the validator by construction. The heuristic counts the objects that
certainly have to move (unmet goals, and everything stacked above them), which
never overestimates, so with weight 1 the plan is a shortest one.

If the time budget ends first, a faster weighted search finishes the job; the
result is then valid but not necessarily shortest.
"""
from __future__ import annotations

import heapq
import time

from .planner import PlanResult
from .scene import Move, Scene, apply, legal_dests


def _key(s: Scene):
    return tuple(sorted(s.on.items()))


def _named(scene: Scene) -> set[str]:
    """Locations the task refers to by name (their identity matters; other free slots are interchangeable)."""
    t = scene.task
    if t is None:
        return set()
    return {a[2] for a in t.atoms if a[0] == "on"} | {x for _, x in t.never_on}


def _successors(s: Scene, named: set[str], keep: frozenset):
    held = s.held()
    movers = [held] if held else sorted(o for o in s.clear() if s.movable(o) and o not in keep)
    blocked = s.blockers
    for o in movers:
        if held is None and any(s.on.get(b) == where for b, where in blocked.get(o, ())):
            continue
        src = s.on[o]
        used_free_slot = False
        for d in legal_dests(s, o):
            if d == src:
                continue
            loc = s.locs.get(d)
            if loc is not None and loc.kind == "buffer" and d not in named:
                if used_free_slot or (src in s.locs and s.locs[src].kind == "buffer" and src not in named):
                    continue                          # free slots are interchangeable
                used_free_slot = True
            yield Move(o, d)


def _h(s: Scene) -> int:
    if s.task is not None:
        need = s.task.must_move(s)
    else:
        goals = s.goals()
        need = set(s.objects) - s.settled(goals)
    held = s.held()
    if held is not None:
        need.add(held)
    return len(need)


def _search(scene: Scene, weight: float, deadline: float, max_expansions: int):
    named = _named(scene)
    keep = scene.task.keep if scene.task is not None else frozenset()
    start = scene.copy()
    counter = 0
    frontier = [(weight * _h(start), 0, counter, start, None, None)]
    best = {_key(start): 0}
    parent: dict = {}
    expansions = 0
    while frontier:
        _, g, _, s, pkey, move = heapq.heappop(frontier)
        key = _key(s)
        if best.get(key, 1 << 30) < g:
            continue
        parent.setdefault(key, (pkey, move))
        if s.goal_reached():
            moves = []
            while parent[key][1] is not None:
                key, m = parent[key]
                moves.append(m)
            return moves[::-1], expansions, ""
        expansions += 1
        if expansions > max_expansions or (expansions % 32 == 0 and time.perf_counter() > deadline):
            return None, expansions, "budget"
        for m in _successors(s, named, keep):
            s2 = s.copy()
            apply(s2, m)
            k2 = _key(s2)
            if best.get(k2, 1 << 30) <= g + 1:
                continue
            best[k2] = g + 1
            counter += 1
            heapq.heappush(frontier, (g + 1 + weight * _h(s2), g + 1, counter, s2, key, m))
    return None, expansions, "exhausted"


def plan_task(scene: Scene, max_ms: float = 300.0, max_expansions: int = 200_000) -> PlanResult:
    """Plan for scene.task (or the default sorting goal). status: ok | partial | infeasible."""
    t0 = time.perf_counter()
    moves, n, why = _search(scene, 1.0, t0 + max_ms * 0.6 / 1e3, max_expansions)
    planner = "task_astar"
    if moves is None and why == "budget":             # finish with a greedier search
        moves, n2, why = _search(scene, 4.0, t0 + max_ms / 1e3, max_expansions)
        n += n2
        planner = "task_weighted_astar"
    if moves is None:
        unmet = "; ".join(scene.unsatisfied()) or "the goal"
        reason = (f"no sequence of legal moves achieves: {unmet}" if why == "exhausted"
                  else f"no plan found within {max_ms:g} ms for: {unmet}")
        res = PlanResult("infeasible", [], planner, reason, n)
    else:
        res = PlanResult("ok", moves, planner, "", n)
    res.deferred = scene.deferred()
    if res.status == "ok" and res.deferred:
        res.status = "partial"
    res.plan_ms = (time.perf_counter() - t0) * 1e3
    return res
