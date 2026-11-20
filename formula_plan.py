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
IMPROVE_WEIGHTS = (10, 5, 3, 2)  # then shorter plans within a budget
# Budgets count arrangements, not seconds: the same request always gets the same answer, however
# busy the machine is. LIMIT_S only guards against a runaway search (then the answer says so).
FIRST_NODES = 40_000
IMPROVE_NODES = 4_000
ORDER_NODES = 100_000            # exact search for ordered goals before falling back to a greedy one
LIMIT_S = 30.0


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


def explain(mover: 'Mover', sup: dict, x: str, dst: str, first: str | None = None) -> str | None:
    """Why may object x NOT go to dst in arrangement sup? None if the move is allowed. The same rules as
    Mover.moves (a test checks that both agree), written out so a rejected step can be explained."""
    items, places = mover.items, mover.places
    if x not in items or x not in sup:
        return f'{x} is not an object of the scene'
    if x in sup.values():
        return f'{x} cannot be picked: {next(y for y, v in sup.items() if v == x)} is on it'
    if x in mover.keep:
        return f'{x} must stay where it is (keep)'
    if first and x != first:
        return f'the first object to move must be {first}'
    if dst == sup[x]:
        return f'{x} is already on {dst}'
    if dst == 'table':
        if mover.no_table or (x, 'table') in mover.never:
            return 'the table may not be used (no_table / never_on)'
        return None
    if dst in items:
        if dst == x:
            return f'{x} cannot go on itself'
        if dst in sup.values() or dst not in sup:
            return f'{dst} is not free: something is on it'
        if dst in mover.no_top:
            return f'nothing can stand on {dst} (no flat top)'
        if (x, dst) in mover.never:
            return f'{x} may never be put on {dst} (never_on)'
        if mover.size_rule and mover.size[x] and mover.size[dst] and mover.size[x] > mover.size[dst] + 1e-6:
            return f'{x} ({mover.size[x]:g} mm) is larger than {dst} ({mover.size[dst]:g} mm)'
        if mover.towers:
            n, height, cur = 1, mover.h[dst], dst
            while sup.get(cur) in items:
                cur = sup[cur]
                n, height = n + 1, height + mover.h[cur]
            root = sup.get(cur)
            if not mover._limit_ok(root, n + 1, height + mover.h[x]):
                return (f'the tower on {root} would be {n + 1} objects / {height + mover.h[x]:g} mm, '
                        f'over its limit')
        return None
    if dst in places:
        d = places[dst]
        if not d.get('open', True):
            return f'{dst} has a closed lid'
        if sum(1 for v in sup.values() if v == dst) >= int(d.get('slots', 1)):
            return f'{dst} is full ({d.get("slots", 1)} slots)'
        if not mover.fits[(x, dst)]:
            return f'{x} is too tall for {dst}'
        if (x, dst) in mover.never:
            return f'{x} may never be put in {dst} (never_on)'
        if mover.towers and not mover._limit_ok(dst, 1, mover.h[x]):
            return f'{x} alone is over the height limit of {dst}'
        return None
    return f'unknown target {dst}'


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


EXACT_NODES = 60_000             # exact phase (plain A*) when a lower-bound estimate is known


def find_moves(sc: dict, goal: goal_logic.Goal, cons: dict | None = None, limit_s: float | None = None,
               heuristic=None, exact=None) -> dict:
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

    stopped = ['done']

    def run(weight, budget, bound, guide=None):
        hf = guide or h
        key = lambda s: tuple(sorted(s.items()))            # noqa: E731
        tie = itertools.count()
        h0 = hf(start)
        pq = [(weight * h0, h0, 0, next(tie), start, [])]
        seen = {key(start): 0}
        used = 0
        while pq:
            f, hv, g, _, sup, path = heapq.heappop(pq)
            expanded[0] += 1
            used += 1
            if g and hv == 0 and done(sup):
                return path
            if used > budget:
                stopped[0] = 'budget'
                return None
            if time.perf_counter() - t0 > limit_s:
                stopped[0] = 'time'
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
                hn = hf(new)
                if hn == 0 and not done(new):
                    hn = 1                                    # the guide says done, the goal does not
                heapq.heappush(pq, (g + 1 + weight * hn, hn, g + 1, next(tie), new, path + [(x, dst)]))
        return None

    if exact is not None:                                    # never overestimates -> the first plan is shortest
        found = run(1, EXACT_NODES, None, guide=exact)
        if found is not None:
            return {'moves': found, 'first': len(found), 'expanded': expanded[0], 'optimal': True,
                    'seconds': round(time.perf_counter() - t0, 3), 'stopped': 'done', 'reason': None}
        if stopped[0] == 'done':                              # searched everything: no plan exists
            return {'moves': None, 'first': None, 'expanded': expanded[0], 'optimal': False, 'stopped': 'exhausted',
                    'seconds': round(time.perf_counter() - t0, 3),
                    'reason': 'no plan exists with these objects, places, limits and constraints (complete search)'}
        stopped[0] = 'done'
    best = run(FIRST_WEIGHT, FIRST_NODES, None)
    out = {'first': len(best) if best else None, 'optimal': False}
    if best:
        for w in IMPROVE_WEIGHTS:
            if stopped[0] == 'time':
                break
            better = run(w, IMPROVE_NODES, len(best))
            if better and len(better) < len(best):
                best = better
    out.update(moves=best, expanded=expanded[0], seconds=round(time.perf_counter() - t0, 3), stopped=stopped[0],
               reason=None if best else f'no sequence of moves reached the goal ({expanded[0]} arrangements tried'
                                        f'{", stopped by the time guard" if stopped[0] == "time" else ""}); '
                                        'it may be impossible with these objects, places and constraints')
    return out


def _fact(f, sup: dict) -> bool:
    return sup.get(f[1]) == ('table' if f[0] == 'on-table' else f[2])


def find_ordered(sc: dict, goal: list, cons: dict | None = None, max_nodes: int | None = None) -> dict:
    """Ordered goals ("first A, then B ..."), searched as ONE task. The state is the arrangement plus how
    many goals are reached. A move may not undo a reached goal and may not meet a later goal before its
    turn (a goal that is already true from the start may wait). A* with an estimate that never
    overestimates (objects the remaining goals force to move): the first plan found is the shortest.
    If the exact search runs out of budget, a greedy search gives a verified, unproven plan.
    -> {'moves', 'optimal', 'stopped' ('done' | 'exhausted' = impossible | 'budget' | 'time'), 'expanded', 'seconds'}"""
    import plan_quality
    t0 = time.perf_counter()
    items, n = sc['items'], len(goal)
    mover = Mover(sc, cons or {})
    first = (cons or {}).get('first')
    first = first[0] if isinstance(first, list) and first else first
    start = dict(sc['support'])

    def advance(sup, k):
        while k < n and _fact(goal[k], sup):
            k += 1
        return k

    def h(sup, k):
        return len(plan_quality.must_move(sup, items, goal[k:])) if k < n else 0

    k0 = advance(start, 0)
    if k0 == n:
        return {'moves': [], 'optimal': True, 'stopped': 'done', 'expanded': 0, 'seconds': 0.0}
    key = lambda s: tuple(sorted(s.items()))                # noqa: E731

    def search(weight, budget):
        tie = itertools.count()
        h0 = h(start, k0)
        pq = [(weight * h0, h0, 0, next(tie), start, k0, [])]
        seen = {(key(start), k0): 0}
        used = 0
        while pq:
            f, hv, g, _, sup, k, path = heapq.heappop(pq)
            if k == n:
                return path, used, 'done'
            used += 1
            if used > budget:
                return None, used, 'budget'
            if time.perf_counter() - t0 > LIMIT_S:
                return None, used, 'time'
            for x, dst in mover.moves(sup, first if g == 0 else None):
                new = dict(sup)
                new[x] = dst
                if any(not _fact(goal[j], new) for j in range(k)):
                    continue                                  # would undo a goal already reached
                if any(_fact(goal[j], new) and not _fact(goal[j], sup) for j in range(k + 1, n)):
                    continue                                  # would meet a later goal before its turn
                k2 = advance(new, k)
                state = (key(new), k2)
                if seen.get(state, 1 << 30) <= g + 1:
                    continue
                seen[state] = g + 1
                hn = h(new, k2)
                heapq.heappush(pq, (g + 1 + weight * hn, hn, g + 1, next(tie), new, k2, path + [(x, dst)]))
        return None, used, 'exhausted'

    path, used, stopped = search(1, max_nodes or ORDER_NODES)
    optimal = path is not None
    if path is None and stopped == 'budget':
        path, more, stopped = search(3, max_nodes or ORDER_NODES)
        used += more
    return {'moves': path, 'optimal': optimal, 'stopped': stopped, 'expanded': used,
            'seconds': round(time.perf_counter() - t0, 3)}


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
