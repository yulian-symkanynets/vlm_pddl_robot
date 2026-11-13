"""Symbolic strategies that need no model: graph planner, A*, PDDL, case base."""
from __future__ import annotations

import tempfile
import time
import warnings
from pathlib import Path

from .base import Proposal, Strategy
from semplan.cases import CaseBase, canonicalize
from semplan.planner import plan, plan_astar
from semplan.scene import HELD, Move, Scene


class GraphStrategy:
    """Scene graph -> constructive planner, A* as complete fallback (architecture H)."""

    def __init__(self, optimal: bool = False, astar_ms: float = 50.0):
        self.name = "astar" if optimal else "graph"
        self.optimal, self.astar_ms = optimal, astar_ms

    def propose(self, scene: Scene) -> Proposal:
        r = plan(scene, optimal=self.optimal, astar_ms=self.astar_ms)
        if r.status == "infeasible":
            return Proposal(None, r.reason, r.planner)
        return Proposal(r.moves, r.reason, r.planner)


class CaseBasedStrategy:
    """Exact structural case retrieval with a fallback strategy (architecture J).

    retrieve: canonical form -> hash lookup;  reuse: bind indices to current ids;
    revise: validate on the current scene;    retain: store the fallback's valid plan.
    """

    def __init__(self, fallback: Strategy | None = None, cases: CaseBase | None = None):
        self.name = "case_based"
        self.fallback = fallback or GraphStrategy()
        self.cases = cases if cases is not None else CaseBase()

    def propose(self, scene: Scene) -> Proposal:
        canon = canonicalize(scene)
        r = self.cases.retrieve(scene, canon)
        if r.outcome == "hit":
            return Proposal(r.moves, source="case")
        if r.outcome == "known_infeasible":
            return Proposal(None, r.reason, "case")
        p = self.fallback.propose(scene)
        if p.moves:
            self.cases.retain(scene, p.moves, p.source or self.fallback.name, canon)
        p.source = f"{r.outcome}->{p.source or self.fallback.name}"
        return p


# ---------------------------------------------------------------- PDDL (architectures C and F)

DOMAIN = """(define (domain sorting)
  (:requirements :strips :typing)
  (:types item slot region)
  (:predicates
    (on ?x - item ?y - item) (at ?x - item ?s - slot) (clear ?x - item)
    (free ?s - slot) (placeable ?s - slot) (slot-in ?s - slot ?r - region)
    (in ?x - item ?r - region) (holding ?x - item) (handempty)
    (supportive ?x - item) (movable ?x - item))
  (:action pick-from-slot
    :parameters (?x - item ?s - slot ?r - region)
    :precondition (and (at ?x ?s) (clear ?x) (handempty) (in ?x ?r) (movable ?x))
    :effect (and (holding ?x) (free ?s) (not (at ?x ?s)) (not (in ?x ?r)) (not (handempty))))
  (:action pick-from-item
    :parameters (?x - item ?y - item ?r - region)
    :precondition (and (on ?x ?y) (clear ?x) (handempty) (in ?x ?r) (movable ?x))
    :effect (and (holding ?x) (clear ?y) (not (on ?x ?y)) (not (in ?x ?r)) (not (handempty))))
  (:action place-on-slot
    :parameters (?x - item ?s - slot ?r - region)
    :precondition (and (holding ?x) (free ?s) (placeable ?s) (slot-in ?s ?r))
    :effect (and (at ?x ?s) (in ?x ?r) (not (free ?s)) (handempty) (not (holding ?x))))
  (:action place-on-item
    :parameters (?x - item ?y - item ?r - region)
    :precondition (and (holding ?x) (clear ?y) (supportive ?y) (in ?y ?r))
    :effect (and (on ?x ?y) (in ?x ?r) (not (clear ?y)) (handempty) (not (holding ?x)))))
"""


def scene_to_pddl(scene: Scene) -> tuple[str, dict[str, str]]:
    """Scene graph -> PDDL problem, by code. Returns (text, slot name -> location id).

    Each ground pose of a zone or free slot becomes a PDDL `slot` object, which is
    how a capacity is expressed in plain STRIPS.
    """
    objs = sorted(scene.objects)
    init, slots, slot_loc, used = [], [], {}, {}
    held = scene.held()
    init.append("(handempty)" if held is None else f"(holding {held})")
    covered = {s for s in scene.on.values() if s in scene.objects}
    for o in objs:
        ob = scene.objects[o]
        if o not in covered:
            init.append(f"(clear {o})")
        if ob.can_support and scene.movable(o):
            init.append(f"(supportive {o})")
        if scene.movable(o):
            init.append(f"(movable {o})")
        s = scene.on[o]
        if s == HELD:
            continue
        init.append(f"(in {o} {scene.base(o)})")
        if s in scene.objects:
            init.append(f"(on {o} {s})")
        else:
            k = used.get(s, 0)
            used[s] = k + 1
            name = f"{s}__{k}"
            slots.append(name)
            slot_loc[name] = s
            init += [f"(at {o} {name})", f"(slot-in {name} {s})"]
            if scene.locs[s].kind != "table":
                init.append(f"(placeable {name})")
    for l in scene.locs.values():
        if l.kind == "table":
            continue
        for k in range(used.get(l.id, 0), l.capacity):
            name = f"{l.id}__{k}"
            slots.append(name)
            slot_loc[name] = l.id
            init += [f"(free {name})", f"(placeable {name})", f"(slot-in {name} {l.id})"]
    goal = [f"(in {o} {z})" for o, z in sorted(scene.goals().items())] + ["(handempty)"]
    text = (f"(define (problem p) (:domain sorting)\n"
            f"  (:objects {' '.join(objs)} - item {' '.join(slots)} - slot "
            f"{' '.join(sorted(scene.locs))} - region)\n"
            f"  (:init {' '.join(init)})\n  (:goal (and {' '.join(goal)})))\n")
    return text, slot_loc


class PddlStrategy:
    """Scene graph -> PDDL by code -> off-the-shelf planner (architecture F).

    engine: "pyperplan" (pure Python, in process), "fast-downward" (C++, one
    subprocess per call) or "fast-downward-opt" (optimal). `last_timing` holds the
    split into generate / parse / solve of the most recent call.
    """

    def __init__(self, engine: str = "pyperplan"):
        from unified_planning.io import PDDLReader
        from unified_planning.shortcuts import OneshotPlanner, get_environment
        get_environment().credits_stream = None
        warnings.filterwarnings("ignore")
        self.name = f"pddl:{engine}"
        self.engine = engine
        self._reader, self._oneshot = PDDLReader, OneshotPlanner
        self._dir = Path(tempfile.mkdtemp(prefix="approaches_pddl_"))
        (self._dir / "domain.pddl").write_text(DOMAIN)
        self.last_timing: dict = {}

    def propose(self, scene: Scene) -> Proposal:
        t0 = time.perf_counter()
        text, slot_loc = scene_to_pddl(scene)
        t1 = time.perf_counter()
        (self._dir / "problem.pddl").write_text(text)
        problem = self._reader().parse_problem(str(self._dir / "domain.pddl"),
                                               str(self._dir / "problem.pddl"))
        t2 = time.perf_counter()
        with self._oneshot(name=self.engine) as planner:
            result = planner.solve(problem)
        t3 = time.perf_counter()
        self.last_timing = {"generate_ms": (t1 - t0) * 1e3, "parse_ms": (t2 - t1) * 1e3,
                            "solve_ms": (t3 - t2) * 1e3}
        if result.plan is None:
            return Proposal(None, f"planner status {result.status.name}", self.engine)
        moves = []
        for a in result.plan.actions:
            args = [str(p) for p in a.actual_parameters]
            if a.action.name == "place-on-slot":
                moves.append(Move(args[0], slot_loc[args[1]]))
            elif a.action.name == "place-on-item":
                moves.append(Move(args[0], args[1]))
        return Proposal(moves, source=self.engine)
