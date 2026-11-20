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
    """Which moves are allowed in a given arrangement: the task's constraints (no_table, never_on,
    keep, first, max_stack_height) and the scene's physical rules (no flat top, clearance, a larger
    object never on a smaller one, tower height in mm and in objects, per place and overall)."""

    def __init__(self, sc: dict, cons: dict):
        import pddl_task
        self.items, self.places = sc['items'], sc['env']['places']
        self.cons = cons or {}
        rules = sc.get('rules') or {}
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
        self.size = {x: d.get('size_mm') for x, d in self.items.items()}
        self.size_rule = bool(rules.get('no_larger_on_smaller'))
        self.h = {x: d.get('height_mm', 0.0) for x, d in self.items.items()}
        self.max_h = rules.get('max_height_mm')
        self.max_n = rules.get('max_objects') or self.cons.get('max_stack_height')
        self.towers = bool(self.max_h or self.max_n or any(
            'max_height_mm' in d or 'max_objects' in d for d in self.places.values()))

    def _limit_ok(self, root, n: int, height: float) -> bool:
        """A tower standing on `root` ("table" or a place) with n objects and this height."""
        d = self.places.get(root, {})
        for lim_n in (self.max_n, d.get('max_objects')):
            if lim_n and n > lim_n:
                return False
        for lim_h in (self.max_h, d.get('max_height_mm')):
            if lim_h and height > lim_h + 1e-6:
                return False
        return True

    def moves(self, sup: dict, only: str | None = None):
        covered = set(sup.values())
        tops = [x for x in self.items if x not in covered and x in sup]
        used = {}
        for v in sup.values():
            if v in self.places:
                used[v] = used.get(v, 0) + 1
        tower = {}                                             # top -> (root, objects, height)
        if self.towers:
            for y in tops:
                n, height, cur = 1, self.h[y], y
                while sup.get(cur) in self.items:
                    cur = sup[cur]
                    n, height = n + 1, height + self.h[cur]
                tower[y] = (sup.get(cur), n, height)
        for x in tops:
            if x in self.keep or (only and x != only):
                continue
            dsts = []
            if not self.no_table and (x, 'table') not in self.never:
                dsts.append('table')
            for y in tops:
                if y == x or y in self.no_top or (x, y) in self.never:
                    continue
                if self.size_rule and self.size[x] and self.size[y] and self.size[x] > self.size[y] + 1e-6:
                    continue
                if self.towers:
                    root, n, height = tower[y]
                    if not self._limit_ok(root, n + 1, height + self.h[x]):
                        continue
                dsts.append(y)
            dsts += [p for p, d in self.places.items()
                     if d.get('open', True) and used.get(p, 0) < int(d.get('slots', 1))
                     and self.fits[(x, p)] and (x, p) not in self.never
                     and (not self.towers or self._limit_ok(p, 1, self.h[x]))]
            for dst in dsts:
                if dst != sup[x]:
                    yield x, dst


def check_moves(sc: dict, moves, cons: dict | None = None) -> str | None:
    """Replay moves with the same rules as the search. -> None, or the first rule a move breaks."""
    mover, sup = Mover(sc, cons or {}), dict(sc['support'])
    first = (cons or {}).get('first')
    first = first[0] if isinstance(first, list) and first else first
    for i, (x, dst) in enumerate(moves):
        if (x, dst) not in set(mover.moves(sup, first if i == 0 else None)):
            return f'move {i + 1} ({x} -> {dst}) breaks a rule of the scene or the task'
        sup[x] = dst
    return None


def find_moves(sc: dict, goal: goal_logic.Goal, cons: dict | None = None, limit_s: float | None = None,
               heuristic=None) -> dict:
    """-> {'moves': [(object, destination)] or None, 'first': n, 'expanded': n, 'seconds': s,
           'reason': text if no plan}.  `heuristic(support) -> int` replaces the default guide
    (requirements of the formula that do not hold yet); the goal test is always the formula."""
    limit_s = LIMIT_S if limit_s is None else limit_s
    t0 = time.perf_counter()
    items = sc['items']
    mover = Mover(sc, cons or {})
    start = dict(sc['support'])
    only = (cons or {}).get('first')
    only = only[0] if isinstance(only, list) and only else only
    expanded = [0]

    def h(sup):
        return heuristic(sup) if heuristic else goal.violated(goal_logic.View(sup, items))

    def done(sup):
        return goal.holds(goal_logic.View(sup, items))

    if done(start):
        return {'moves': [], 'first': 0, 'expanded': 0, 'seconds': 0.0, 'reason': None}

    def run(weight, deadline, bound):
        key = lambda s: tuple(sorted(s.items()))            # noqa: E731
        tie = itertools.count()
        h0 = h(start)
        pq = [(weight * h0, h0, 0, next(tie), start, [])]
        seen = {key(start): 0}
        while pq:
            f, hv, g, _, sup, path = heapq.heappop(pq)
            expanded[0] += 1
            if g and hv == 0 and done(sup):
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
                hn = h(new)
                if hn == 0 and not done(new):
                    hn = 1                                    # the guide says done, the goal does not
                heapq.heappush(pq, (g + 1 + weight * hn, hn, g + 1, next(tie), new, path + [(x, dst)]))
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
