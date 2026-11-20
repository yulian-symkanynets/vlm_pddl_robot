"""Find moves for ANY goal formula (goal_logic.py): weighted A* over arrangements.

State: what every object rests on ("table", another object, or a place). A move takes one object
that has nothing on it and puts it on the table, on another free object, or into a place with a
free slot. The goal is a test (the formula holds), and the guide is the number of requirements
of the formula that do not hold yet.

That guide can be too high, so the first plan is not proven shortest. The search therefore starts
greedy (weight 30), then tries again with lower weights and keeps the shortest plan found. Measured
on the 16-object "every stack one colour" scene: 18 moves in under 0.1 s, against 15 for the
exact colour_stacks.py search and no plan within 60 s from the general PDDL planner.

Constraints of the task are respected: no_table, never_on, keep, first.
"""
from __future__ import annotations

import heapq
import itertools
import time

import goal_logic
from scenes import slot_names

FIRST_WEIGHT = 30                # greedy: a first plan fast, even with 50+ objects
IMPROVE_WEIGHTS = (10, 5, 3, 2)  # then shorter plans while time allows
IMPROVE_ROUND_S = 0.4
LIMIT_S = 8.0                    # an impossible goal is only given up after this long


class Mover:
    """Which moves are allowed in a given arrangement, under the task's constraints."""

    def __init__(self, sc: dict, cons: dict):
        import pddl_task
        self.items, self.places = sc['items'], sc['env']['places']
        self.cons = cons or {}
        self.keep = set(self.cons.get('keep') or [])
        self.no_top = {x for x, d in self.items.items() if pddl_task._no_top(d)}
        self.never = set()
        for a, t in self.cons.get('never_on') or []:
            for x in (self.items if a == '*' else [a]):
                self.never.add((x, t))
        self.no_table = bool(self.cons.get('no_table')) or ('*', 'table') in {(a, t) for a, t in
                                                                              self.cons.get('never_on') or []}
        self.fits = {}
        for p, d in self.places.items():
            c = d.get('clearance_mm')
            for x, i in self.items.items():
                self.fits[(x, p)] = c is None or i.get('height_mm', 0) <= c

    def moves(self, sup: dict, only: str | None = None):
        covered = set(sup.values())
        tops = [x for x in self.items if x not in covered]
        used = {}
        for v in sup.values():
            if v in self.places:
                used[v] = used.get(v, 0) + 1
        for x in tops:
            if x in self.keep or (only and x != only):
                continue
            dsts = []
            if not self.no_table and (x, 'table') not in self.never:
                dsts.append('table')
            dsts += [y for y in tops if y != x and y not in self.no_top and (x, y) not in self.never]
            dsts += [p for p, d in self.places.items()
                     if d.get('open', True) and used.get(p, 0) < int(d.get('slots', 1))
                     and self.fits[(x, p)] and (x, p) not in self.never]
            for dst in dsts:
                if dst != sup[x]:
                    yield x, dst


def find_moves(sc: dict, goal: goal_logic.Goal, cons: dict | None = None, limit_s: float | None = None) -> dict:
    """-> {'moves': [(object, destination)] or None, 'first': n, 'expanded': n, 'seconds': s,
           'reason': text if no plan}"""
    limit_s = LIMIT_S if limit_s is None else limit_s
    t0 = time.perf_counter()
    items = sc['items']
    mover = Mover(sc, cons or {})
    start = dict(sc['support'])
    only = (cons or {}).get('first')
    only = only[0] if isinstance(only, list) and only else only
    expanded = [0]

    def h(sup):
        return goal.violated(goal_logic.View(sup, items))

    if h(start) == 0:
        return {'moves': [], 'first': 0, 'expanded': 0, 'seconds': 0.0, 'reason': None}

    def run(weight, deadline, bound):
        key = lambda s: tuple(sorted(s.items()))            # noqa: E731
        tie = itertools.count()
        pq = [(weight * h(start), 0, next(tie), start, [])]
        seen = {key(start): 0}
        while pq:
            f, g, _, sup, path = heapq.heappop(pq)
            expanded[0] += 1
            if g and h(sup) == 0:
                return path
            if time.perf_counter() > deadline:
                return None
            if bound is not None and g + 1 >= bound:
                continue
            for x, dst in mover.moves(sup, only if g == 0 else None):
                new = dict(sup)
                new[x] = dst
                k = key(new)
                if seen.get(k, 1 << 30) <= g + 1:
                    continue
                seen[k] = g + 1
                hv = h(new)
                heapq.heappush(pq, (g + 1 + weight * hv, g + 1, next(tie), new, path + [(x, dst)]))
        return None

    best = run(FIRST_WEIGHT, t0 + limit_s, None)
    out = {'first': len(best) if best else None}
    if best:
        for w in IMPROVE_WEIGHTS:
            if time.perf_counter() - t0 > limit_s:
                break
            better = run(w, min(t0 + limit_s, time.perf_counter() + IMPROVE_ROUND_S), len(best))
            if better and len(better) < len(best):
                best = better
    out.update(moves=best, expanded=expanded[0], seconds=round(time.perf_counter() - t0, 3),
               reason=None if best else f'no sequence of moves reached the goal within {limit_s:g} s '
                                        f'({expanded[0]} arrangements tried); it may be impossible '
                                        'with these objects, places and constraints')
    return out


def to_actions(moves, sc: dict) -> list[tuple[str, list[str]]]:
    """Moves -> the domain's actions, choosing slots like scenes.state_facts does."""
    places = sc['env']['places']
    cur = dict(sc['support'])
    used, slot_of = {}, {}
    for x, s in cur.items():                                 # same slot choice as the problem file
        if s in places:
            free = [n for n in slot_names(s, int(places[s].get('slots', 1))) if n not in used]
            used[free[0]], slot_of[x] = x, free[0]
    acts = []
    for x, dst in moves:
        src = cur[x]
        if src == 'table':
            acts.append(('pick-from-table', [x]))
        elif src in places:
            acts.append(('pick-from-place', [x, slot_of[x], src]))
            del used[slot_of.pop(x)]
        else:
            acts.append(('pick-from-item', [x, src]))
        if dst == 'table':
            acts.append(('place-on-table', [x]))
        elif dst in places:
            s = next(n for n in slot_names(dst, int(places[dst].get('slots', 1))) if n not in used)
            used[s], slot_of[x] = x, s
            acts.append(('place-in', [x, s, dst]))
        else:
            acts.append(('place-on-item', [x, dst]))
        cur[x] = dst
    return acts
