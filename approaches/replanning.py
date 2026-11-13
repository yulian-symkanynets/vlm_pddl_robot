"""Three ways to react when the observed scene no longer matches the plan.

All three take the same inputs (the newly observed scene and the moves that were
still to be executed) and return moves that the shared validator accepts:

  FullReplan           throw the old plan away, plan from the observed scene
  ValidateThenReplan   keep the old plan if it still validates, else full replan
  PrefixRepair         keep the longest still-executable prefix of the old plan,
                       plan the rest from where that prefix ends (plan repair)

A fourth mode, receding horizon, is an execution policy rather than a repair
method: plan, execute only the first step, observe, plan again (see
`receding_horizon_episode`). The behaviour-tree variant is in behavior_tree.py.

None of them assumes that an action succeeded: the input is always the scene the
laptop observed, with its version number.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

from .base import Strategy
from .symbolic import GraphStrategy
from semplan.scene import Move, Scene, apply, validate, why_not


@dataclass
class Replan:
    moves: list[Move] | None
    decision: str                 # kept | replanned | repaired | failed
    reused: int = 0               # old moves carried over unchanged
    ms: float = 0.0
    based_on_version: int = -1


class FullReplan:
    name = "full_replan"

    def __init__(self, planner: Strategy | None = None):
        self.planner = planner or GraphStrategy()

    def replan(self, observed: Scene, remaining: list[Move]) -> Replan:
        t0 = time.perf_counter()
        p = self.planner.propose(observed)
        ok = p.moves is not None and validate(observed, p.moves).ok
        old = set(remaining)
        return Replan(p.moves if ok else None, "replanned" if ok else "failed",
                      sum(m in old for m in p.moves or []), (time.perf_counter() - t0) * 1e3,
                      observed.version)


class ValidateThenReplan(FullReplan):
    """The recommended default: one cheap replay decides whether anything has to change."""
    name = "validate_then_replan"

    def replan(self, observed: Scene, remaining: list[Move]) -> Replan:
        t0 = time.perf_counter()
        if remaining and validate(observed, remaining).ok:
            return Replan(list(remaining), "kept", len(remaining),
                          (time.perf_counter() - t0) * 1e3, observed.version)
        r = super().replan(observed, remaining)
        r.ms = (time.perf_counter() - t0) * 1e3
        return r


class PrefixRepair(FullReplan):
    """Plan repair: reuse what still works, plan only the rest."""
    name = "prefix_repair"

    def replan(self, observed: Scene, remaining: list[Move]) -> Replan:
        t0 = time.perf_counter()
        s, kept = observed.copy(), []
        for m in remaining:
            if why_not(s, m):
                break
            apply(s, m)
            kept.append(m)
            if s.goal_reached():
                break
        p = self.planner.propose(s)
        moves = kept + p.moves if p.moves is not None else None
        ok = moves is not None and validate(observed, moves).ok
        decision = "failed" if not ok else "kept" if len(kept) == len(moves) else "repaired"
        return Replan(moves if ok else None, decision, len(kept),
                      (time.perf_counter() - t0) * 1e3, observed.version)


def receding_horizon_episode(robot, planner: Strategy | None = None, max_steps: int = 200) -> dict:
    """Plan -> execute ONE move -> observe -> plan again. `robot` is a semplan.sim.SimRobot.

    Nothing is ever reused, so nothing can be stale; the cost is one planning call
    per executed move, which only makes sense when planning is cheap.
    """
    from semplan.scene import parse_scene, to_contract
    planner = planner or GraphStrategy()
    plans = steps = 0
    for _ in range(max_steps):
        scene = parse_scene(robot.observe())
        p = planner.propose(scene)
        plans += 1
        if not p.moves:
            break
        for step in to_contract(scene, p.moves[:1]):         # PICK + PLACE of the first move
            steps += 1
            if robot.execute(step)["status"] != "success":
                break                                        # observe again and re-plan
    final = robot.truth
    return {"goal": final.goal_reached(), "plans": plans, "robot_steps": steps}


def compare(n: int = 300, seed: int = 0, sizes=(4, 8)) -> list[dict]:
    """Disturb random plans half-way and run the three repair methods on the same cases."""
    from semplan.gen import random_scene
    from semplan.planner import plan, quick_infeasible
    from semplan.scene import dump_scene, parse_scene
    from semplan.sim import SimRobot
    rng = random.Random(seed)
    methods = [FullReplan(), ValidateThenReplan(), PrefixRepair()]
    rows = {m.name: {"ms": [], "moves": [], "reused": [], "kept": 0, "failed": 0} for m in methods}
    cases = 0
    while cases < n:
        s = random_scene(rng, rng.randint(*sizes), n_colors=3, n_buffers=2)
        if quick_infeasible(s):
            continue
        r = plan(s)
        if r.status != "ok" or len(r.moves) < 2:
            continue
        k = rng.randrange(1, len(r.moves))
        cur = s.copy()
        for m in r.moves[:k]:
            apply(cur, m)
        robot = SimRobot(cur, rng)
        robot.human_disturbance()
        observed = parse_scene(dict(dump_scene(robot.truth), scene_version=2))
        if quick_infeasible(observed):
            continue
        cases += 1
        for m in methods:
            out = m.replan(observed, r.moves[k:])
            row = rows[m.name]
            if out.moves is None:
                row["failed"] += 1
                continue
            row["ms"].append(out.ms)
            row["moves"].append(len(out.moves))
            row["reused"].append(out.reused)
            row["kept"] += out.decision == "kept"
    table = []
    for name, row in rows.items():
        ms = sorted(row["ms"])
        table.append({"method": name, "cases": cases, "failed": row["failed"], "kept": row["kept"],
                      "p50_ms": ms[len(ms) // 2], "p95_ms": ms[int(.95 * len(ms))],
                      "mean_moves": sum(row["moves"]) / len(row["moves"]),
                      "mean_reused": sum(row["reused"]) / len(row["reused"])})
    return table
