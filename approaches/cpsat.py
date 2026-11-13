"""Planning as constraint solving (OR-Tools CP-SAT), with side constraints.

The plan is not searched for step by step; it is described. For a horizon of T
steps there is one Boolean per (object, time, support): "object o rests on s at
time t". Constraints say what a legal sequence of scenes looks like, the goal is
a constraint on the last scene, and the solver minimises the number of changes.

    exactly one support per object and time
    at most one object on each object; nothing on a prism; zone and slot capacity
    at most one object changes its support between t and t+1   (one move per step)
    an object that changes must have nothing on it
    nothing is ever put on the open table
    at time T every goal object is in a stack that stands in its zone

The reason to use this instead of the graph planner is not speed. It is that
extra requirements are one line each, and the solver still returns a provably
shortest plan that respects them:

    forbid_on=[("blue_cylinder_1", "red_cube_1")]   never place that on this
    move_once={"glass_cube"}                         fragile: at most one move
    avoid=["tmp_2"]                                  do not use this slot

Needs `pip install ortools`.
"""
from __future__ import annotations

from .base import Proposal, Strategy
from .symbolic import GraphStrategy
from semplan.scene import HELD, Move, Scene


class CpSatStrategy:
    def __init__(self, forbid_on: list[tuple[str, str]] | None = None,
                 move_once: set[str] | None = None, avoid: list[str] | None = None,
                 max_horizon: int = 14, time_limit_s: float = 5.0,
                 upper_bound: Strategy | None = None):
        self.name = "cp_sat"
        self.forbid_on = forbid_on or []
        self.move_once = move_once or set()
        self.avoid = set(avoid or [])
        self.max_horizon, self.time_limit_s = max_horizon, time_limit_s
        self.upper_bound = upper_bound or GraphStrategy()
        self.last_stats: dict = {}

    def propose(self, scene: Scene) -> Proposal:
        from ortools.sat.python import cp_model
        if scene.held() is not None:
            return Proposal(None, "CP-SAT model expects an empty gripper", "cp_sat")
        objs = sorted(scene.objects)
        locs = sorted(scene.locs)
        sup = objs + locs
        goals = scene.goals()
        # The constructive plan gives a horizon that is certainly long enough when no
        # side constraints are active; with side constraints allow some slack.
        ub = self.upper_bound.propose(scene)
        slack = 3 if (self.forbid_on or self.move_once or self.avoid) else 0
        T = min(self.max_horizon, (len(ub.moves) if ub.moves is not None else 2 * len(objs)) + slack)
        if T == 0:
            return Proposal([], source="cp_sat")
        m = cp_model.CpModel()
        x = {(o, t, s): m.NewBoolVar(f"x_{o}_{t}_{s}") for o in objs for t in range(T + 1)
             for s in sup if s != o}
        ch = {(o, t): m.NewBoolVar(f"ch_{o}_{t}") for o in objs for t in range(T)}
        for o in objs:
            for t in range(T + 1):
                m.AddExactlyOne(x[o, t, s] for s in sup if s != o)
            for s in sup:
                if s != o:
                    m.Add(x[o, 0, s] == int(scene.on[o] == s))            # initial scene
        for t in range(T + 1):
            for s in objs:                                               # one object per top
                on_s = [x[o, t, s] for o in objs if o != s]
                if not scene.objects[s].can_support or not scene.movable(s):
                    for o in objs:
                        if o != s and scene.on[o] != s:
                            m.Add(x[o, t, s] == 0)
                else:
                    m.AddAtMostOne(on_s)
            for l in locs:                                               # capacity
                if scene.locs[l].kind != "table":
                    cap = 0 if l in self.avoid and scene.floor_used(l) == 0 else scene.locs[l].capacity
                    m.Add(sum(x[o, t, l] for o in objs) <= cap)
        for t in range(T):
            m.AddAtMostOne(ch[o, t] for o in objs)                       # one move per step
            for o in objs:
                for s in sup:
                    if s != o:                                           # ch = 1 if the support changes
                        m.Add(ch[o, t] >= x[o, t, s] - x[o, t + 1, s])
                for y in objs:
                    if y != o:                                           # the mover must be clear
                        m.Add(ch[o, t] + x[y, t, o] <= 1)
                if not scene.movable(o):
                    m.Add(ch[o, t] == 0)
                for l in locs:
                    if scene.locs[l].kind == "table":                    # never onto the table
                        m.Add(x[o, t + 1, l] <= x[o, t, l])
        # goal: region of o at time T. r[o, l] = "o's stack stands in location l"
        r = {(o, l): m.NewBoolVar(f"r_{o}_{l}") for o in objs for l in locs}
        for o in objs:
            m.AddExactlyOne(r[o, l] for l in locs)
            for l in locs:
                m.AddImplication(x[o, T, l], r[o, l])
            for s in objs:
                if s != o:
                    for l in locs:
                        m.Add(r[o, l] == r[s, l]).OnlyEnforceIf(x[o, T, s])
            if o in goals:
                m.Add(r[o, goals[o]] == 1)
        # side constraints
        for a, b in self.forbid_on:
            if a in scene.objects and b in scene.objects:
                for t in range(1, T + 1):
                    m.Add(x[a, t, b] <= x[a, 0, b])                      # may stay, may not be put there
        for o in self.move_once & set(objs):
            m.Add(sum(ch[o, t] for t in range(T)) <= 1)
        m.Minimize(sum(ch.values()))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_s
        solver.parameters.num_workers = 1                                # deterministic, comparable timing
        status = solver.Solve(m)
        self.last_stats = {"horizon": T, "booleans": len(x) + len(ch) + len(r),
                           "status": solver.StatusName(status), "optimal": status == cp_model.OPTIMAL}
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return Proposal(None, f"solver: {solver.StatusName(status)} within horizon {T}", "cp_sat")
        moves = []
        for t in range(T):
            for o in objs:
                if solver.Value(ch[o, t]):
                    dst = next(s for s in sup if s != o and solver.Value(x[o, t + 1, s]))
                    moves.append(Move(o, dst))
        return Proposal(moves, source=f"cp_sat {solver.StatusName(status).lower()}")
