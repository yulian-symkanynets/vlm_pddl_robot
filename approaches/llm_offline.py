"""The model works OFFLINE; at run time only deterministic code executes.

  synthesize_policy   the model writes a Python policy for the whole domain; it is
                      tested on training scenes and the failures are fed back
                      (generalized planning with LLMs, Silver et al. 2024)
  ProgramStrategy     runs a saved policy: no model call per request (architecture L)

  synthesize_heuristic  the model writes heuristic functions; the best one on the
                        training scenes is kept (Correa, Pereira and Seipp 2025)
  HeuristicSearchStrategy  greedy best-first search with a saved heuristic

Both write plain Python to approaches/generated/, where it can be read, reviewed
and version-controlled like any other code.

Security: generated code is executed with exec(). The namespace is restricted,
but that is not a sandbox. Review generated files before using them, and run
synthesis only with a model and prompts you control.
"""
from __future__ import annotations

import heapq
from pathlib import Path

from .base import Proposal
from .llm import LLM, extract_code
from semplan.scene import Move, Scene, apply, legal_dests, why_not

GENERATED = Path(__file__).parent / "generated"

API_DOC = """You are given `v`, a read-only view of the scene, with these methods:
  v.objects()            list of object ids
  v.goal(o)              zone id the object must end in, or None
  v.on(o)                what the object rests on: an object id or a location id
  v.region(o)            location at the bottom of the object's stack
  v.is_clear(o)          True if nothing is on top of the object
  v.can_support(o)       True if another object may be placed on it
  v.done(o)              True if the object is final: in its goal zone and everything under it is final
  v.zones()              list of zone ids
  v.free_slots()         list of free temporary slot ids
  v.has_room(location)   True if the zone or slot has a free ground pose
  v.legal_destinations(o)  every place the object may be put right now
"""

POLICY_PROMPT = """Write a Python function for a robot that sorts objects into zones.

def policy(v):
    \"\"\"Return the next action as (object_id, destination) or None when finished.\"\"\"

""" + API_DOC + """
One action moves one clear object onto a legal destination.
The function is called again after every action, on the new scene.
It must work for any number of objects, stacks and zones.
{feedback}Answer with one ```python block containing only the function."""

HEURISTIC_PROMPT = """Write a Python heuristic for greedy best-first search in a sorting domain.

def h(v):
    \"\"\"Estimate the number of moves still needed. Lower is better. 0 only at the goal.\"\"\"

""" + API_DOC + """
Answer with one ```python block containing only the function. Variant {i} of {n}: try a
different idea than the obvious one."""


class View:
    """The small, read-only API that generated code is allowed to use."""

    def __init__(self, scene: Scene):
        self._s = scene
        self._goals = scene.goals()
        self._settled = scene.settled(self._goals)
        self._clear = scene.clear()

    def objects(self): return sorted(self._s.objects)
    def goal(self, o): return self._goals.get(o)
    def on(self, o): return self._s.on[o]
    def region(self, o): return self._s.base(o)
    def is_clear(self, o): return o in self._clear
    def can_support(self, o): return self._s.objects[o].can_support
    def done(self, o): return o in self._settled
    def zones(self): return sorted(l.id for l in self._s.locs.values() if l.kind == "zone")
    def free_slots(self):
        return sorted(l.id for l in self._s.locs.values()
                      if l.kind == "buffer" and self._s.floor_free(l.id) > 0)
    def has_room(self, loc): return loc in self._s.locs and self._s.locs[loc].kind != "table" \
        and self._s.floor_free(loc) > 0
    def legal_destinations(self, o):
        return [d for d in legal_dests(self._s, o) if d != self._s.on[o]]


SAFE_BUILTINS = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
                 for k in ("len", "range", "sorted", "min", "max", "sum", "any", "all", "list",
                           "dict", "set", "tuple", "enumerate", "zip", "abs", "bool", "int",
                           "float", "str", "isinstance", "reversed", "next", "iter", "map", "filter")}


def load_function(code: str, name: str):
    """Compile generated code and return the function `name` from it."""
    ns: dict = {"__builtins__": SAFE_BUILTINS}
    exec(compile(code, f"<generated {name}>", "exec"), ns)
    if name not in ns or not callable(ns[name]):
        raise ValueError(f"the code does not define {name}()")
    return ns[name]


# ================================================================ policy programs

def rollout(policy, scene: Scene, max_steps: int | None = None) -> tuple[list[Move], str]:
    """Run a policy until it returns None. -> (moves, error message or '')."""
    s = scene.copy()
    moves: list[Move] = []
    seen = {tuple(sorted(s.on.items()))}
    for _ in range(max_steps or 4 * len(s.objects) + 8):
        try:
            act = policy(View(s))
        except Exception as e:                               # generated code may do anything
            return moves, f"policy raised {type(e).__name__}: {e}"
        if act is None:
            return moves, "" if s.goal_reached() else \
                f"policy returned None but these objects are not in their zone: {s.unsatisfied()}"
        try:
            m = Move(str(act[0]), str(act[1]))
        except Exception:
            return moves, f"policy returned {act!r}, expected (object, destination) or None"
        reason = why_not(s, m)
        if reason:
            return moves, f"illegal action {m}: {reason}"
        apply(s, m)
        moves.append(m)
        key = tuple(sorted(s.on.items()))
        if key in seen:
            return moves, f"policy went back to an earlier state after {m} (endless loop)"
        seen.add(key)
    return moves, "step limit reached without finishing"


def describe(scene: Scene) -> str:
    stacks = "; ".join(f"on {b}: {' < '.join(col)} (bottom to top)" for b, col in scene.stacks())
    zones = ", ".join(f"{l.id} has {scene.floor_free(l.id)} free ground pose(s)"
                      for l in scene.locs.values() if l.kind == "zone")
    return f"stacks: {stacks}. goals: {scene.goals()}. {zones}. free slots: {View(scene).free_slots()}"


def synthesize_policy(llm: LLM, train: list[Scene], max_rounds: int = 4, accept: float = 1.0,
                      save_as: str | None = "policy.py") -> tuple[str | None, list[dict]]:
    """Ask for a policy, test it on training scenes, feed back the first failure, repeat.

    Returns (code or None, log). The best version is returned if it solves at least
    the fraction `accept` of the training scenes. A policy below 1.0 is still
    usable behind the validator, with another strategy as fallback for the scenes
    it gets wrong. Passing the training scenes says nothing about other scenes.
    """
    feedback, log, best = "", [], (0, None, 0)
    for round_no in range(1, max_rounds + 1):
        code = extract_code(llm.complete(POLICY_PROMPT.format(feedback=feedback)))
        try:
            policy = load_function(code, "policy")
        except Exception as e:
            err = f"the code could not be loaded: {type(e).__name__}: {e}"
            log.append({"round": round_no, "solved": 0, "of": len(train), "error": err})
            feedback = f"Your previous code:\n{code}\nProblem: {err}\n"
            continue
        failures = [(s, err) for s in train if (err := rollout(policy, s)[1])]
        solved = len(train) - len(failures)
        log.append({"round": round_no, "solved": solved, "of": len(train),
                    "error": failures[0][1] if failures else ""})
        if solved > best[0]:
            best = (solved, code, round_no)
        if not failures:
            break
        s, err = failures[0]
        feedback = (f"Your previous code:\n{code}\nIt failed on {len(failures)} of {len(train)} "
                    f"test scenes. First failure: {err}\nScene: {describe(s)}\n")
    solved, code, round_no = best
    if code is None or solved < accept * len(train):
        return None, log
    if save_as:
        GENERATED.mkdir(exist_ok=True)
        (GENERATED / save_as).write_text(
            f'"""Generated policy. Solved {solved}/{len(train)} training scenes '
            f'(round {round_no}). Review before use."""\n\n{code}\n')
    return code, log


class ProgramStrategy:
    """Runs a synthesized policy. Zero model calls per request."""

    def __init__(self, code: str | None = None, path: str | Path | None = None):
        self.name = "llm_program"
        code = code if code is not None else Path(path or GENERATED / "policy.py").read_text()
        self.policy = load_function(code, "policy")

    def propose(self, scene: Scene) -> Proposal:
        moves, err = rollout(self.policy, scene)
        if err:
            return Proposal(None, err, "program")
        return Proposal(moves, source="program")


# ================================================================ heuristic search

def best_first(scene: Scene, h, max_expansions: int = 20_000) -> tuple[list[Move] | None, int]:
    """Greedy best-first search over scenes with heuristic h(View). -> (moves, expansions)."""
    start = scene.copy()
    counter = 0
    frontier = [(h(View(start)), counter, start, [])]
    seen = {tuple(sorted(start.on.items()))}
    expansions = 0
    while frontier:
        _, _, s, path = heapq.heappop(frontier)
        if s.goal_reached():
            return path, expansions
        expansions += 1
        if expansions > max_expansions:
            return None, expansions
        held = s.held()
        movers = [held] if held else sorted(o for o in s.clear() if s.movable(o))
        for o in movers:
            for d in legal_dests(s, o):
                if d == s.on[o]:
                    continue
                s2 = s.copy()
                apply(s2, Move(o, d))
                key = tuple(sorted(s2.on.items()))
                if key in seen:
                    continue
                seen.add(key)
                counter += 1
                heapq.heappush(frontier, (h(View(s2)), counter, s2, path + [Move(o, d)]))
    return None, expansions


def synthesize_heuristic(llm: LLM, train: list[Scene], n_candidates: int = 3,
                         save_as: str | None = "heuristic.py") -> tuple[str | None, list[dict]]:
    """Ask for several heuristics, keep the one that solves most training scenes with
    the fewest expansions."""
    best, log = None, []
    for i in range(1, n_candidates + 1):
        code = extract_code(llm.complete(HEURISTIC_PROMPT.format(i=i, n=n_candidates)))
        try:
            h = load_function(code, "h")
            results = [best_first(s, h, 5_000) for s in train]
        except Exception as e:
            log.append({"candidate": i, "error": f"{type(e).__name__}: {e}"})
            continue
        solved = sum(m is not None for m, _ in results)
        expansions = sum(e for _, e in results)
        log.append({"candidate": i, "solved": solved, "of": len(train), "expansions": expansions})
        score = (-solved, expansions)
        if best is None or score < best[0]:
            best = (score, code, i)
    if best is None:
        return None, log
    if save_as:
        GENERATED.mkdir(exist_ok=True)
        (GENERATED / save_as).write_text(
            f'"""Generated heuristic (candidate {best[2]}). Review before use."""\n\n{best[1]}\n')
    return best[1], log


class HeuristicSearchStrategy:
    """Greedy best-first search with a generated (or hand-written) heuristic."""

    def __init__(self, code: str | None = None, path: str | Path | None = None,
                 max_expansions: int = 20_000):
        self.name = "llm_heuristic"
        code = code if code is not None else Path(path or GENERATED / "heuristic.py").read_text()
        self.h = load_function(code, "h")
        self.max_expansions = max_expansions

    def propose(self, scene: Scene) -> Proposal:
        moves, n = best_first(scene, self.h, self.max_expansions)
        if moves is None:
            return Proposal(None, f"search budget exhausted after {n} expansions", "gbfs")
        return Proposal(moves, source=f"gbfs ({n} expansions)")
