"""Experience memory: structural case base, plan adaptation, execution statistics.

The key idea is a *canonical form*: a complete description of the planning-relevant
structure of a scene in which object ids, colour names and stack order have been
replaced by canonical indices. Two scenes with the same canonical form are
isomorphic as planning problems, so a plan stored in canonical indices transfers
by substituting the current scene's objects for the indices. The key is not a
lossy hash or an embedding, so a hit can never be a "similar but different" scene.

A looser, similarity-based retriever (feature vector + nearest neighbour) is also
provided, mainly so that its value can be measured against the exact one.
"""
from __future__ import annotations

import itertools
import json
import sqlite3
import time
from dataclasses import dataclass, field

from .planner import PlanResult, plan
from .scene import HELD, Move, Scene, apply, validate, why_not

MAX_PERMS = 24      # zone orderings tried when zone signatures tie


# ================================================================ canonical form

@dataclass
class Canon:
    key: tuple                       # hashable canonical form
    objs: list[str]                  # canonical object index -> object id in this scene
    zones: list[str]                 # canonical zone index  -> zone id in this scene
    free_buffers: list[str]          # free buffer ids, sorted
    tokens: list[tuple]              # per canonical object: (goal, can_support, movable, settled)

    def n_unsettled(self) -> int:
        return sum(1 for t in self.tokens if not t[3])


def canonicalize(scene: Scene) -> Canon:
    goals = scene.goals()
    settled = scene.settled(goals)
    zones = sorted(l.id for l in scene.locs.values() if l.kind == "zone")
    # Reduced stacks: drop settled objects that are buried under other settled objects.
    raw = []                                     # (base kind, base zone or None, [objects])
    for base, col in scene.stacks():
        first = next((i for i, o in enumerate(col) if o not in settled), len(col))
        keep_from = max(first - 1, 0) if first < len(col) else len(col) - 1
        kept = col[keep_from:]
        if base == HELD:
            raw.append(("held", None, kept))
            continue
        loc = scene.locs[base]
        elevated = keep_from > 0
        kind = ("e" if elevated else "") + loc.kind      # "ezone": on a buried settled object
        raw.append((kind, base if loc.kind == "zone" else None, kept))
    free_floor = {z: scene.floor_free(z) for z in zones}

    def tok(o, zi):
        ob = scene.objects[o]
        g = goals.get(o)
        return (zi[g] if g is not None else -1, ob.can_support, scene.movable(o), o in settled)

    def encode(order):
        zi = {z: i for i, z in enumerate(order)}
        st = sorted(((kind, zi[bz] if bz is not None else -1, tuple(tok(o, zi) for o in col)), col)
                    for kind, bz, col in raw)
        return (tuple(free_floor[z] for z in order), tuple(e for e, _ in st)), st

    # Order zones by a relabelling-invariant signature, then break ties by trying
    # the permutations inside each tied group and keeping the smallest encoding.
    def sig(z):
        inside = sorted((kind, tuple((goals.get(o) == z, goals.get(o) is None,
                                      scene.objects[o].can_support, scene.movable(o),
                                      o in settled) for o in col))
                        for kind, bz, col in raw if bz == z)
        wanted = sorted((kind, bz == z, i, len(col) - i, scene.objects[o].can_support)
                        for kind, bz, col in raw for i, o in enumerate(col) if goals.get(o) == z)
        return (free_floor[z], tuple(inside), tuple(wanted))

    sigs = {z: sig(z) for z in zones}
    groups = [list(g) for _, g in itertools.groupby(sorted(zones, key=lambda z: sigs[z]),
                                                    key=lambda z: sigs[z])]
    n_perms = 1
    for g in groups:
        for k in range(2, len(g) + 1):
            n_perms *= k
    if n_perms == 1 or n_perms > MAX_PERMS:
        orders = [[z for g in groups for z in g]]
    else:
        orders = ([z for g in combo for z in g]
                  for combo in itertools.product(*(itertools.permutations(g) for g in groups)))
    best = None
    for order in orders:
        key, st = encode(order)
        if best is None or key < best[0]:
            best = (key, st, order)
    key, st, order = best
    objs = [o for _, col in st for o in col]
    zi = {z: i for i, z in enumerate(order)}
    free_buffers = sorted(l.id for l in scene.locs.values()
                          if l.kind == "buffer" and scene.floor_free(l.id) > 0)
    return Canon(key, objs, list(order), free_buffers, [tok(o, zi) for o in objs])


# ================================================================ plans in canonical terms

def abstract_plan(scene: Scene, canon: Canon, moves: list[Move]) -> tuple[list, int] | None:
    """Rewrite a concrete plan in canonical indices. None if it cannot be expressed."""
    oi = {o: i for i, o in enumerate(canon.objs)}
    zi = {z: i for i, z in enumerate(canon.zones)}
    buf_order: list[str] = []
    occupied = {scene.on[o]: o for o in scene.objects
                if scene.on[o] in scene.locs and scene.locs[scene.on[o]].kind == "buffer"}
    steps = []
    for m in moves:
        if m.obj not in oi:
            return None
        if m.dst in oi:
            d = ("o", oi[m.dst])
        elif m.dst in zi:
            d = ("z", zi[m.dst])
        elif m.dst in occupied and occupied[m.dst] in oi:
            d = ("s", oi[occupied[m.dst]])        # the buffer this object stood on at the start
        elif m.dst in canon.free_buffers:
            if m.dst not in buf_order:
                buf_order.append(m.dst)
            d = ("b", buf_order.index(m.dst))
        else:
            return None
        steps.append((oi[m.obj], d))
    return steps, len(buf_order)


def bind_plan(scene: Scene, canon: Canon, steps: list) -> list[Move] | None:
    """Substitute the current scene's objects into a canonical plan."""
    out = []
    try:
        for o, (kind, i) in steps:
            if kind == "o":
                d = canon.objs[i]
            elif kind == "z":
                d = canon.zones[i]
            elif kind == "b":
                d = canon.free_buffers[i]
            else:
                d = scene.on[canon.objs[i]]
            out.append(Move(canon.objs[o], d))
    except IndexError:
        return None
    return out


# ================================================================ case base

@dataclass
class Case:
    steps: list                      # [(obj index, (dest kind, dest index))]
    buffers: int                     # free buffers the plan needs
    source: str                      # planner that produced it
    uses: int = 0
    exec_ok: int = 0
    exec_fail: int = 0
    created: float = field(default_factory=time.time)
    last_used: float = 0.0
    optimal: bool = False

    def reliability(self) -> float:
        """Posterior mean of execution success (Beta(1,1) prior)."""
        return (self.exec_ok + 1) / (self.exec_ok + self.exec_fail + 2)


@dataclass
class Retrieval:
    moves: list[Move] | None
    outcome: str                     # "hit" | "miss" | "rejected"
    case: Case | None = None
    key: tuple | None = None
    reason: str = ""


class CaseBase:
    """Exact structural case base: canonical form -> validated plans."""

    def __init__(self, min_reliability: float = 0.34, max_variants: int = 3):
        self.cases: dict[tuple, list[Case]] = {}
        self.min_reliability = min_reliability
        self.max_variants = max_variants
        self.stats = {"hit": 0, "miss": 0, "rejected": 0, "stored": 0, "known_infeasible": 0}
        # Negative cases: canonical form -> (free slots it was unsolvable with, reason, proven)
        self.dead: dict[tuple, tuple[int, str, bool]] = {}

    def __len__(self) -> int:
        return sum(len(v) for v in self.cases.values())

    def retrieve(self, scene: Scene, canon: Canon | None = None) -> Retrieval:
        """Retrieve -> reuse (bind) -> revise (validate). Never returns an unvalidated plan."""
        canon = canon or canonicalize(scene)
        variants = self.cases.get(canon.key)
        dead = self.dead.get(canon.key)
        if dead and len(canon.free_buffers) <= dead[0] and not variants:
            self.stats["known_infeasible"] += 1
            return Retrieval(None, "known_infeasible", key=canon.key, reason=dead[1])
        if not variants:
            self.stats["miss"] += 1
            return Retrieval(None, "miss", key=canon.key)
        reason = "no variant fits"
        for case in sorted(variants, key=lambda c: (len(c.steps), -c.reliability())):
            if case.buffers > len(canon.free_buffers):
                reason = "not enough free slots for the stored plan"
                continue
            if case.reliability() < self.min_reliability:
                reason = "stored plan has a poor execution record"
                continue
            moves = bind_plan(scene, canon, case.steps)
            if moves is None:
                continue
            v = validate(scene, moves)
            if not v.ok:
                reason = v.reason
                continue
            case.uses += 1
            case.last_used = time.time()
            self.stats["hit"] += 1
            return Retrieval(moves, "hit", case, canon.key)
        self.stats["rejected"] += 1
        return Retrieval(None, "rejected", key=canon.key, reason=reason)

    def retain(self, scene: Scene, moves: list[Move], source: str,
               canon: Canon | None = None, optimal: bool = False) -> Case | None:
        """Store a plan that has passed validation. Keeps the best few variants per key."""
        canon = canon or canonicalize(scene)
        ab = abstract_plan(scene, canon, moves)
        if ab is None:
            return None
        steps, nbuf = ab
        variants = self.cases.setdefault(canon.key, [])
        for c in variants:
            if c.buffers <= nbuf and len(c.steps) <= len(steps):
                return c                                   # dominated: nothing new to learn
        case = Case(steps, nbuf, source, optimal=optimal)
        variants[:] = [c for c in variants
                       if not (c.buffers >= nbuf and len(c.steps) >= len(steps))]
        variants.append(case)
        variants.sort(key=lambda c: (len(c.steps), c.buffers))
        del variants[self.max_variants:]
        self.stats["stored"] += 1
        return case

    def retain_infeasible(self, canon: Canon, reason: str, proven: bool) -> None:
        """Remember that this structure had no plan with this many free slots."""
        old = self.dead.get(canon.key)
        n = len(canon.free_buffers)
        if old is None or n > old[0] or (proven and not old[2]):
            self.dead[canon.key] = (max(n, old[0]) if old else n, reason, proven)

    def evict(self, max_cases: int) -> int:
        """Bound memory: drop the least recently used keys."""
        if len(self.cases) <= max_cases:
            return 0
        order = sorted(self.cases, key=lambda k: max(c.last_used or c.created for c in self.cases[k]))
        drop = order[:len(self.cases) - max_cases]
        for k in drop:
            del self.cases[k]
        return len(drop)

    # ------------------------------------------------------------ persistence (sqlite, stdlib)
    def save(self, path: str) -> None:
        db = sqlite3.connect(path)
        db.execute("CREATE TABLE IF NOT EXISTS cases (key TEXT PRIMARY KEY, variants TEXT)")
        db.execute("DELETE FROM cases")
        db.executemany("INSERT INTO cases VALUES (?, ?)", (
            (repr(k), json.dumps([c.__dict__ for c in v])) for k, v in self.cases.items()))
        db.commit()
        db.close()

    @classmethod
    def load(cls, path: str, **kw) -> "CaseBase":
        import ast
        cb = cls(**kw)
        db = sqlite3.connect(path)
        for k, v in db.execute("SELECT key, variants FROM cases"):
            variants = []
            for d in json.loads(v):
                d["steps"] = [(o, tuple(dst)) for o, dst in d["steps"]]
                variants.append(Case(**d))
            cb.cases[ast.literal_eval(k)] = variants
        db.close()
        return cb


# ================================================================ similarity retrieval + repair

def features(scene: Scene, canon: Canon | None = None) -> list[float]:
    """Hand-made structural feature vector (relabelling invariant)."""
    goals = scene.goals()
    settled = scene.settled(goals)
    stacks = [(b, c) for b, c in scene.stacks() if b != HELD]
    heights = [len(c) for _, c in stacks]
    uns = [o for o in scene.objects if o not in settled]
    wrong_zone = sum(1 for b, c in stacks if scene.locs[b].kind == "zone"
                     for o in c if goals.get(o) not in (None, b))
    buried = sum(1 for _, c in stacks for o in c[:-1] if o not in settled and o in goals)
    zones = [l for l in scene.locs.values() if l.kind == "zone"]
    f = [len(scene.objects), len(uns), len(stacks), max(heights, default=0),
         sum(1 for h in heights if h == 1), sum(1 for h in heights if h == 2),
         sum(1 for h in heights if h >= 3), wrong_zone, buried,
         sum(1 for o in uns if not scene.objects[o].can_support),
         sum(1 for l in scene.locs.values() if l.kind == "buffer" and scene.floor_free(l.id) > 0),
         len(zones), sum(scene.floor_free(z.id) for z in zones),
         sum(1 for z in zones if scene.floor_free(z.id) == 0),
         1.0 if scene.held() else 0.0, len(scene.deferred())]
    return [float(x) for x in f]


def repair(scene: Scene, candidate: list[Move]) -> tuple[PlanResult, int]:
    """Keep the longest executable prefix of a candidate plan, then plan the rest.

    Returns (result, number of candidate steps reused).
    """
    s = scene.copy()
    kept: list[Move] = []
    for m in candidate:
        if why_not(s, m):
            break
        apply(s, m)
        kept.append(m)
        if s.goal_reached():
            break
    rest = plan(s)
    rest.moves = kept + rest.moves
    rest.planner = f"repair({len(kept)})+{rest.planner}"
    return rest, len(kept)


class SimilarityIndex:
    """Nearest-neighbour case retrieval over feature vectors (brute force, numpy)."""

    def __init__(self):
        import numpy as np
        self.np = np
        self.vecs: list[list[float]] = []
        self.payload: list[tuple[list, list]] = []      # (canonical steps, tokens)
        self._mat = None

    def add(self, scene: Scene, canon: Canon, moves: list[Move]) -> None:
        ab = abstract_plan(scene, canon, moves)
        if ab is None:
            return
        self.vecs.append(features(scene, canon))
        self.payload.append((ab[0], canon.tokens))
        self._mat = None

    def nearest(self, scene: Scene, canon: Canon):
        if not self.vecs:
            return None, float("inf")
        if self._mat is None:
            self._mat = self.np.asarray(self.vecs, dtype=self.np.float32)
        q = self.np.asarray(features(scene, canon), dtype=self.np.float32)
        d = ((self._mat - q) ** 2).sum(axis=1)
        i = int(d.argmin())
        return self.payload[i], float(d[i])

    def adapt(self, scene: Scene, canon: Canon, payload) -> list[Move]:
        """Map roles position by position; steps touching an unmatched role are dropped."""
        steps, tokens = payload
        ok = [i < len(canon.tokens) and canon.tokens[i][:3] == t[:3] for i, t in enumerate(tokens)]
        usable = []
        for o, (kind, i) in steps:
            if o >= len(ok) or not ok[o]:
                continue
            if kind in ("o", "s") and (i >= len(ok) or not ok[i]):
                continue
            usable.append((o, (kind, i)))
        return bind_plan(scene, canon, usable) or []


# ================================================================ execution statistics

class ActionStats:
    """Success statistics per abstract action context, learned from execution feedback.

    This is knowledge the symbolic model cannot derive: e.g. that picking a prism
    off a cylinder fails far more often than picking a cube off the table.
    """

    def __init__(self):
        self.counts: dict[tuple, list[int]] = {}

    @staticmethod
    def context(scene: Scene, action: str, obj: str, other: str | None) -> tuple:
        def kind(x):
            if x is None or x == HELD:
                return "none"
            return scene.objects[x].type if x in scene.objects else scene.locs[x].kind
        return (action, scene.objects[obj].type if obj in scene.objects else "?", kind(other))

    def record(self, ctx: tuple, success: bool) -> None:
        c = self.counts.setdefault(ctx, [0, 0])
        c[0 if success else 1] += 1

    def p_success(self, ctx: tuple) -> float:
        ok, fail = self.counts.get(ctx, (0, 0))
        return (ok + 1) / (ok + fail + 2)

    def dump(self) -> dict:
        return {"|".join(k): {"ok": v[0], "fail": v[1], "p": round(self.p_success(k), 3)}
                for k, v in sorted(self.counts.items())}
