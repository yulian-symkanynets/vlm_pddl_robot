"""Goal "one stack per colour": dedicated shortest-plan search.

The general PDDL planner handles this goal badly (any object of a colour may be the base and the
others may come in any order, so its estimates are weak: 32 moves fast, no optimum in 2 min for
16 objects). This A* search works on the stacks directly:

  state     the stacks on the table, bottom to top
  move      the top object of a stack goes onto the table, or onto a stack that is one colour
            and has that object's colour (parking on another object is never better than the
            table, which has room for everything)
  estimate  objects that are not yet in a one-colour run at the bottom of their stack must move;
            of several such runs of one colour, all but the largest must be taken apart.
            Every such object needs at least one move, so the estimate never overestimates and
            the first plan found is the shortest.
  goal      every stack is one colour, one stack per colour

Scope: objects on the table or on each other (not in zones), no hidden objects, no constraints;
anything else falls back to the PDDL planner.
"""
from __future__ import annotations

import heapq
import itertools
import time


def stacks_of(support: dict[str, str], items) -> list[tuple[str, ...]] | None:
    """Support map -> stacks bottom to top, or None if something is not on the table or an object."""
    above = {}
    for x, y in support.items():
        if y != "table" and y not in items:
            return None
        if y in items:
            above[y] = x
    out = []
    for x, y in support.items():
        if y == "table":
            st = [x]
            while st[-1] in above:
                st.append(above[st[-1]])
            out.append(tuple(st))
    return out if sum(map(len, out)) == len(support) else None


def _key(stacks):
    return tuple(sorted(stacks))


def _estimate(stacks, colour) -> int:
    best, total, unsettled = {}, {}, 0
    for st in stacks:
        c, k = colour[st[0]], 1
        while k < len(st) and colour[st[k]] == c:
            k += 1
        unsettled += len(st) - k
        best[c] = max(best.get(c, 0), k)
        total[c] = total.get(c, 0) + k
    return unsettled + sum(total[c] - best[c] for c in total)


def solve(support: dict[str, str], colour: dict[str, str], time_limit_s: float = 10.0):
    """-> (moves [(object, from, to)], optimal) or (None, False). from/to: object id or "table"."""
    start = stacks_of(support, colour)
    if start is None:
        return None, False
    t0, tie = time.perf_counter(), itertools.count()
    frontier = [(_estimate(start, colour), 0, next(tie), start, [])]
    seen = {_key(start): 0}
    while frontier:
        f, g, _, stacks, path = heapq.heappop(frontier)
        h = f - g
        if h == 0:
            return path, True
        if time.perf_counter() - t0 > time_limit_s:
            return None, False
        for i, st in enumerate(stacks):
            x, rest = st[-1], st[:-1]
            src = rest[-1] if rest else "table"
            targets = [] if not rest else [("table", None)]
            targets += [(t[-1], j) for j, t in enumerate(stacks)
                        if j != i and all(colour[y] == colour[x] for y in t)]
            for dst, j in targets:
                new = [s for k, s in enumerate(stacks) if k not in (i, j)]
                if rest:
                    new.append(rest)
                new.append((x,) if j is None else stacks[j] + (x,))
                k = _key(new)
                if seen.get(k, 1 << 30) <= g + 1:
                    continue
                seen[k] = g + 1
                heapq.heappush(frontier, (g + 1 + _estimate(new, colour), g + 1, next(tie), new,
                                          path + [(x, src, dst)]))
    return None, False


def to_actions(moves, final_support: dict[str, str] | None = None) -> list[tuple[str, list[str]]]:
    """Moves -> PDDL actions of the base domain (pick-from-table/-item, place-on-table/-item)."""
    acts = []
    for x, src, dst in moves:
        acts.append(("pick-from-table", [x]) if src == "table" else ("pick-from-item", [x, src]))
        acts.append(("place-on-table", [x]) if dst == "table" else ("place-on-item", [x, dst]))
    return acts
