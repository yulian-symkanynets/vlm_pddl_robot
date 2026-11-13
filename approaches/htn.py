"""A small Hierarchical Task Network planner (architecture G).

Total-order HTN in the style of SHOP / GTPyhop: a task is either a primitive
action (a Move) or is decomposed by a method into subtasks. Planning is
depth-first: decompose the first task, apply primitives to a copy of the state,
backtrack when a method does not apply.

The domain knowledge is written by hand in the methods below. That is the point
of HTN: the planner is only as complete as its methods, but it does exactly what
the methods say and nothing else.
"""
from __future__ import annotations

from .base import Proposal
from semplan.scene import Move, Scene, apply, why_not

MAX_DEPTH = 60


# ---------------------------------------------------------------- helpers on the state

def _home(s: Scene, o: str, goals: dict) -> str | None:
    """A final destination that is available right now: zone floor or a finished stack."""
    z = goals.get(o)
    if z is None:
        return None
    settled, clear = s.settled(goals), s.clear()
    tops = sorted(t for t in clear if t != o and t in settled and s.objects[t].can_support
                  and s.movable(t) and s.base(t) == z)
    if not s.objects[o].can_support and tops:
        return tops[0]                       # a prism ends a stack: prefer an existing top
    if s.floor_free(z) > 0:
        return z
    return tops[0] if tops else None


def _parking(s: Scene, o: str) -> list[str]:
    """Places to put an object that is in the way, best first."""
    out = sorted(l.id for l in s.locs.values() if l.kind == "buffer" and s.floor_free(l.id) > 0)
    out += sorted(l.id for l in s.locs.values()
                  if l.kind == "zone" and s.floor_free(l.id) > 0 and l.id != s.on[o])
    return out


def _above(s: Scene, o: str) -> str | None:
    return s.above().get(o)


# ---------------------------------------------------------------- methods
# A method returns a list of alternative decompositions (each a list of tasks).

def m_sort_all(s: Scene, goals: dict):
    todo = [o for o in sorted(goals) if s.base(o) != goals[o] or o not in s.settled(goals)]
    if not todo:
        return [[]]
    # try objects that can go home immediately first, then the rest
    ready = [o for o in todo if o in s.clear() and _home(s, o, goals)]
    order = ready + [o for o in todo if o not in ready]
    return [[("deliver", o), ("sort_all",)] for o in order]


def m_deliver(s: Scene, goals: dict, o: str):
    if o in s.settled(goals):
        return [[]]
    top = _above(s, o)
    if top is not None:                                  # something is on it: uncover first
        return [[("get_out_of_the_way", top), ("deliver", o)]]
    home = _home(s, o, goals)
    if home is not None:
        return [[("move", o, home)]]
    z = goals[o]                                         # zone is full: empty one ground pose
    blockers = sorted(b for b, sup in s.on.items() if sup == z and b not in s.settled(goals))
    return [[("get_out_of_the_way", b), ("deliver", o)] for b in blockers]


def m_get_out_of_the_way(s: Scene, goals: dict, o: str):
    top = _above(s, o)
    if top is not None:
        return [[("get_out_of_the_way", top), ("get_out_of_the_way", o)]]
    options = []
    home = _home(s, o, goals)
    if home is not None:
        options.append([("move", o, home)])              # best: it goes straight home
    options += [[("move", o, d)] for d in _parking(s, o)]
    return options


METHODS = {"sort_all": m_sort_all, "deliver": m_deliver,
           "get_out_of_the_way": m_get_out_of_the_way}


# ---------------------------------------------------------------- the planner

def seek_plan(s: Scene, tasks: list, goals: dict, depth: int = 0, seen: set | None = None):
    """Depth-first HTN search. Returns a list of Move or None."""
    if not tasks:
        return [] if s.goal_reached() else None
    if depth > MAX_DEPTH:
        return None
    task, rest = tasks[0], tasks[1:]
    if task[0] == "move":                                # primitive task
        m = Move(task[1], task[2])
        if why_not(s, m):
            return None
        s2 = s.copy()
        apply(s2, m)
        key = (tuple(sorted(s2.on.items())), tuple(rest))
        seen = set() if seen is None else seen
        if key in seen:                                  # same state, same agenda: a loop
            return None
        seen.add(key)
        tail = seek_plan(s2, rest, goals, depth + 1, seen)
        return None if tail is None else [m] + tail
    for subtasks in METHODS[task[0]](s, goals, *task[1:]):
        plan = seek_plan(s, list(subtasks) + rest, goals, depth + 1, seen)
        if plan is not None:
            return plan
    return None


class HtnStrategy:
    name = "htn"

    def propose(self, scene: Scene) -> Proposal:
        held = scene.held()
        tasks = [("sort_all",)]
        if held is not None:                             # put the held object down first
            tasks = [("get_out_of_the_way", held)] + tasks
        try:
            moves = seek_plan(scene, tasks, scene.goals())
        except RecursionError:
            moves = None
        if moves is None:
            return Proposal(None, "no decomposition found", "htn")
        return Proposal(moves, source="htn")
