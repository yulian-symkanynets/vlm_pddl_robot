"""Task planners for the sorting domain.

plan_greedy   polynomial-time constructive planner over the support graph
plan_astar    optimal A* search, used as fallback and as the optimality reference
plan          the production entry point: greedy first, A* only if greedy is stuck

Both work on the Move(obj, dst) action model of scene.py and return moves that
have not yet been validated; the caller validates before anything is executed.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

from .scene import HELD, Move, Scene, apply, why_not_place

INF = 10 ** 9


@dataclass
class PlanResult:
    status: str                     # "ok" | "partial" | "infeasible"
    moves: list[Move] = field(default_factory=list)
    planner: str = ""
    reason: str = ""
    expansions: int = 0
    plan_ms: float = 0.0
    deferred: dict[str, str] = field(default_factory=dict)   # object -> why it was left out


# ================================================================ greedy constructive planner

def _pedestals(s: Scene, zone: str, settled: set[str], clear: set[str], exclude: str) -> list[str]:
    """Settled clear tops inside `zone` that another object may be stacked on."""
    return sorted(o for o in clear
                  if o != exclude and o in settled and s.objects[o].can_support
                  and s.movable(o) and s.base(o) == zone)


def _constructive_dest(s: Scene, o: str, zone: str, settled: set[str], clear: set[str],
                       pending: dict[str, int], force: bool = False) -> str | None:
    """A destination from which `o` never has to move again, or None.

    `pending[zone]` counts the supporting (flat-top) objects still headed for the zone.
    A supporting object keeps the number of usable spots constant (it takes one and
    becomes one); a non-supporting object (prism) uses one up. So a prism may only
    be placed if a spot remains for the supporting objects that still have to come.
    """
    peds = _pedestals(s, zone, settled, clear, o)
    free = s.floor_free(zone)
    if s.objects[o].can_support:
        if free > 0:
            return zone
        return peds[0] if peds else None
    if not force and pending.get(zone, 0) > 0 and free + len(peds) < 2:
        return None
    if peds:
        return peds[0]                          # keep ground poses for objects that can carry
    return zone if free > 0 else None


def _dig_score(s: Scene, o: str, goals: dict[str, str], settled: set[str],
               clear: set[str], pending: dict[str, int]) -> int:
    """How many buffer moves, starting with `o`, until something useful is exposed."""
    if s.locs.get(s.on[o]) is not None and s.locs[s.on[o]].kind == "buffer":
        return INF                              # already parked
    j, below = 1, s.on[o]
    while below in s.objects:
        if below in settled:                    # exposes a pedestal
            z = s.base(below)
            useful = s.locs[z].kind == "zone" and pending.get(z, 0) > 0 \
                and s.objects[below].can_support
            return j if useful else INF
        z = goals.get(below)
        if z is not None and (s.floor_free(z) > 0 or _pedestals(s, z, settled, clear, below)):
            return j                            # `below` can go home once uncovered
        j += 1
        below = s.on[below]
    loc = s.locs[below]
    if loc.kind == "zone" and pending.get(loc.id, 0) > 0:
        return j                                # frees a ground pose somebody needs
    return INF


def _park_dest(s: Scene, o: str, settled: set[str], clear: set[str],
               pending: dict[str, int]) -> str | None:
    """Where to put an object that has to get out of the way."""
    for l in sorted(s.locs.values(), key=lambda l: l.id):
        if l.kind == "buffer" and s.floor_free(l.id) > 0:
            return l.id
    column = set()
    x = o
    while x in s.objects:
        column.add(x)
        x = s.on[x]
    best, best_rank = None, None
    for t in sorted(clear):
        if t in column or why_not_place(s, o, t):
            continue
        base = s.locs[s.base(t)]
        if base.kind == "buffer":
            rank = 0                            # pile parked objects together
        elif t in settled and not (base.kind == "zone" and pending.get(base.id, 0) > 0):
            rank = 1                            # a finished stack nobody is waiting for
        elif t in settled:
            rank = 4                            # blocks a pedestal that is in demand
        else:
            rank = 3                            # an object that still has to move itself
        if best_rank is None or rank < best_rank:
            best, best_rank = t, rank
    spare = sorted((pending.get(l.id, 0) - s.floor_free(l.id), l.id) for l in s.locs.values()
                   if l.kind == "zone" and s.floor_free(l.id) > 0 and l.id != s.on[o])
    if spare and (best_rank is None or best_rank > 2):
        return spare[0][1]                      # rank 2: borrow a ground pose of another zone
    return best


def plan_greedy(scene: Scene) -> PlanResult:
    """Constructive planner.

    Repeat: (1) if some clear object can go straight to a final position, move it;
    (2) otherwise park the object whose removal exposes something useful soonest.
    With enough free slots every object moves at most twice.
    """
    s = scene.copy()
    goals = s.goals()
    moves: list[Move] = []
    seen = {tuple(sorted(s.on.items()))}
    for _ in range(4 * len(s.objects) + 8):
        settled = s.settled(goals)
        clear = s.clear()
        held = s.held()
        todo = [o for o, z in goals.items() if s.base(o) != z]
        if not todo and held is None:
            return PlanResult("ok", moves, "greedy")
        pending: dict[str, int] = {}             # zone -> supporting objects still to arrive
        for o, z in goals.items():
            if o not in settled and s.objects[o].can_support:
                pending[z] = pending.get(z, 0) + 1
        cands = [held] if held else sorted(
            (o for o in clear if o not in settled and s.movable(o)),
            key=lambda o: (not s.objects[o].can_support, o))
        move = None
        for o in cands:                                         # (1) constructive move
            z = goals.get(o)
            if z is None:
                continue
            d = _constructive_dest(s, o, z, settled, clear, pending)
            if d is not None:
                move = Move(o, d)
                break
        if move is None and held is not None:                   # must put it down somewhere
            d = _park_dest(s, held, settled, clear, pending)
            move = Move(held, d) if d else None
        elif move is None:                                      # (2) park a blocker
            scored = sorted((_dig_score(s, o, goals, settled, clear, pending), o) for o in cands)
            for sc, o in scored:
                if sc >= INF:
                    break
                d = _park_dest(s, o, settled, clear, pending)
                if d is not None:
                    move = Move(o, d)
                    break
            if move is None:                                    # (3) evict a finished prism that
                for o in sorted(clear & settled):               #     blocks the last usable spot
                    z = goals.get(o)
                    if z and not s.objects[o].can_support and pending.get(z, 0) > 0 \
                            and s.floor_free(z) == 0 and s.movable(o) \
                            and not _pedestals(s, z, settled, clear, o):
                        d = _park_dest(s, o, settled, clear, pending)
                        if d is not None:
                            move = Move(o, d)
                            break
            if move is None:                                    # last resort: forced placement
                for o in cands:
                    z = goals.get(o)
                    d = z and _constructive_dest(s, o, z, settled, clear, pending, force=True)
                    if d:
                        move = Move(o, d)
                        break
        if move is None:
            return PlanResult("infeasible", moves, "greedy", "greedy planner is stuck")
        apply(s, move)
        moves.append(move)
        key = tuple(sorted(s.on.items()))
        if key in seen:
            return PlanResult("infeasible", moves, "greedy", "greedy planner cycled")
        seen.add(key)
    return PlanResult("infeasible", moves, "greedy", "greedy step limit reached")


# ================================================================ optimal A* search

def plan_astar(scene: Scene, max_expansions: int = 200_000,
               max_ms: float | None = None) -> PlanResult:
    """Optimal planner (fewest moves). State = tuple of supports, encoded as ints.

    Heuristic: number of objects that are not settled. Every such object has to be
    moved at least once, so the heuristic is admissible and the result optimal.
    Free buffers are interchangeable, so only the first free one is expanded.
    """
    ids = sorted(scene.objects)
    n = len(ids)
    idx = {o: i for i, o in enumerate(ids)}
    loc_ids = sorted(l for l in scene.locs)
    lidx = {l: n + i for i, l in enumerate(loc_ids)}
    kind = {lidx[l]: scene.locs[l].kind for l in loc_ids}
    cap = {lidx[l]: scene.locs[l].capacity for l in loc_ids}
    buffers = [lidx[l] for l in loc_ids if scene.locs[l].kind == "buffer"]
    places = [lidx[l] for l in loc_ids if scene.locs[l].kind == "zone"]
    goals_named = scene.goals()
    goal = [lidx[goals_named[o]] if o in goals_named else -2 for o in ids]
    sup_ok = [scene.objects[o].can_support and scene.movable(o) for o in ids]
    mov_ok = [scene.movable(o) for o in ids]
    H = -1                                           # gripper

    def enc(x: str) -> int:
        return H if x == HELD else idx[x] if x in idx else lidx[x]

    start = tuple(enc(scene.on[o]) for o in ids)

    def analyse(st):
        """-> (bases, unsettled count, goal reached)"""
        base = [0] * n
        ok = [0] * n                                  # 0 unknown, 1 settled, 2 unsettled
        for i in range(n):
            chain, x = [], i
            while True:                               # walk down to a known node / location
                if x == H or x >= n or ok[x]:
                    break
                chain.append(x)
                x = st[x]
            if x == H:
                b, below_ok = H, False
            elif x >= n:
                b, below_ok = x, True
            else:
                b, below_ok = base[x], ok[x] == 1
            for y in reversed(chain):
                base[y] = b
                below_ok = below_ok and (goal[y] == -2 or goal[y] == b)
                ok[y] = 1 if below_ok else 2
        uns = sum(1 for v in ok if v == 2)
        done = all(goal[i] == -2 or base[i] == goal[i] for i in range(n)) and H not in st
        return uns, done

    h0, done0 = analyse(start)
    if done0:
        return PlanResult("ok", [], "astar")
    counter = 0
    deadline = None if max_ms is None else time.perf_counter() + max_ms / 1e3
    openq = [(h0, 0, counter, start)]
    best_g = {start: 0}
    parent: dict = {start: None}
    expansions = 0
    while openq:
        f, negg, _, st = heapq.heappop(openq)
        g = -negg
        if best_g.get(st, INF) < g:
            continue
        uns, done = analyse(st)
        if done:
            moves = []
            while parent[st] is not None:
                prev, (o, d) = parent[st]
                moves.append(Move(ids[o], ids[d] if d < n else loc_ids[d - n]))
                st = prev
            return PlanResult("ok", moves[::-1], "astar", expansions=expansions)
        expansions += 1
        if expansions > max_expansions or (
                deadline is not None and expansions % 64 == 0 and time.perf_counter() > deadline):
            return PlanResult("infeasible", [], "astar", "search budget exhausted", expansions)
        covered = {x for x in st if 0 <= x < n}
        used: dict[int, int] = {}
        for x in st:
            if x >= n:
                used[x] = used.get(x, 0) + 1
        held = st.index(H) if H in st else None
        movers = [held] if held is not None else \
            [i for i in range(n) if i not in covered and mov_ok[i]]
        dests = [z for z in places if used.get(z, 0) < cap[z]]
        fb = next((b for b in buffers if used.get(b, 0) == 0), None)
        if fb is not None:
            dests.append(fb)
        tops = [i for i in range(n) if i not in covered and sup_ok[i] and st[i] != H]
        for o in movers:
            if not mov_ok[o]:
                continue
            for d in dests + tops:
                if d == o or d == st[o]:
                    continue
                if d >= n and kind[d] == "buffer" and st[o] >= n and kind[st[o]] == "buffer":
                    continue                          # buffer -> buffer is never useful
                nxt = st[:o] + (d,) + st[o + 1:]
                ng = g + 1
                if best_g.get(nxt, INF) <= ng:
                    continue
                best_g[nxt] = ng
                parent[nxt] = (st, (o, d))
                counter += 1
                heapq.heappush(openq, (ng + analyse(nxt)[0], -ng, counter, nxt))
    return PlanResult("infeasible", [], "astar", "no plan exists for this scene", expansions)


# ================================================================ entry point

def quick_infeasible(scene: Scene) -> str | None:
    """Cheap necessary conditions. A reason string means no plan can exist.

    In a finished zone every ground pose carries one stack and a stack can end in
    at most one non-supporting object, so a zone needs at least as many usable
    ground poses as it has prisms to receive (and at least one if anything goes there).
    """
    goals = scene.goals()
    frozen = scene.frozen()
    need: dict[str, list[int]] = {}
    for o, z in goals.items():
        c = need.setdefault(z, [0, 0])
        c[0] += 1
        c[1] += not scene.objects[o].can_support
    for z, (total, prisms) in need.items():
        blocked = sum(1 for o, s in scene.on.items() if s == z and o in frozen)
        usable = scene.locs[z].capacity - blocked
        if usable < max(prisms, 1):
            return (f"{z} offers {usable} usable ground pose(s) but must receive {total} "
                    f"object(s), {prisms} of which cannot carry another object")
    return None


def plan(scene: Scene, optimal: bool = False, astar_budget: int = 200_000,
         astar_ms: float | None = None) -> PlanResult:
    """Production planner: greedy, with A* as a complete fallback.

    Objects the server may not plan for (unidentified, unverified pose, unreachable,
    or underneath such an object) are left where they are and listed in `deferred`;
    the status is then "partial" and the caller should request more information.
    `astar_ms` bounds the fallback by wall-clock time, which keeps latency predictable.
    """
    t0 = time.perf_counter()
    why = quick_infeasible(scene)
    if why:
        res = PlanResult("infeasible", [], "precheck", why)
    elif optimal:
        res = plan_astar(scene, astar_budget, astar_ms)
    else:
        res = plan_greedy(scene)
        if res.status != "ok":
            fallback = plan_astar(scene, astar_budget, astar_ms)
            fallback.planner = "greedy->astar"
            res = fallback
    res.deferred = scene.deferred()
    if res.status == "ok" and res.deferred:
        res.status = "partial"
    res.plan_ms = (time.perf_counter() - t0) * 1e3
    return res
