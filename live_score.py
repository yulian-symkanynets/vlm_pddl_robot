"""Scoring for the live benchmark (live_bench.py).

Every judgement here is made independently of the server's own checks:

    replay            runs a plan step by step on the GROUND-TRUTH scene with its own rules
                      (hand empty, object free, target free, room in the zone, lid open)
    satisfies         evaluates a human-written reference formula on the final arrangement
    check_plan        the "_check" block of examples/requests/*.json (on / in / max_moves / ...)
    perception_score  compares a scene the server read from a photo with the true scene
    judge             combines them into one outcome per request

Plain data in, plain data out: no server and no GPU are needed, so tests/test_live_bench.py
runs these offline.
"""
from __future__ import annotations

import re

import goal_logic
import openworld

OUTCOMES = ('correct', 'wrong plan', 'asked', 'no plan', 'unexpected ok', 'error')


def _sc(scene: dict) -> dict:
    return openworld.normalize_scene(scene)


# --------------------------------------------------------------------------- replay
def replay(scene: dict, plan: list[dict]) -> dict:
    """Execute `plan` ([{"action", "args"}]) on `scene`. -> ok, error, final support, picks, placed."""
    sc = _sc(scene)
    items, places = sc['items'], sc['env']['places']
    sup, held = dict(sc['support']), None
    opened = {p for p, d in places.items() if d.get('open', True)}
    picks, placed = [], []

    def fail(i, msg):
        return {'ok': False, 'error': f'step {i + 1}: {msg}', 'support': sup, 'picks': picks, 'placed': placed}

    def free(x):
        return x != held and x not in sup.values()

    for i, step in enumerate(plan or []):
        a, args = step.get('action', ''), list(step.get('args') or [])
        if a == 'observe':
            continue
        if a == 'open-lid':
            opened.add(args[0] if args else '')
            continue
        if not args:
            return fail(i, f'{a} without arguments')
        x = args[0]
        if a.startswith('pick-'):
            if held:
                return fail(i, f'hand is not empty (holding {held})')
            if x not in sup:
                return fail(i, f'unknown object {x}')
            if not free(x):
                return fail(i, f'{x} has something on it')
            if (a == 'pick-from-table' and sup[x] != 'table') or \
                    (a == 'pick-from-item' and sup[x] != args[1]) or \
                    (a == 'pick-from-place' and sup[x] != args[-1]):
                return fail(i, f'{x} is on {sup[x]}, the plan says otherwise')
            picks.append(x)
            held = x
            del sup[x]
        elif a.startswith('place-'):
            if held != x:
                return fail(i, f'{x} is not in the hand (holding {held})')
            if a == 'place-on-table':
                sup[x] = 'table'
            elif a == 'place-on-item':
                y = args[1] if len(args) > 1 else None
                if y not in items or y == x or y not in sup or not free(y):
                    return fail(i, f'{y} is not a free object to put {x} on')
                sup[x] = y
            elif a == 'place-in':
                p = args[-1]
                if p not in places:
                    return fail(i, f'unknown place {p}')
                if p not in opened:
                    return fail(i, f'{p} has a closed lid')
                if sum(1 for v in sup.values() if v == p) >= int(places[p].get('slots', 1)):
                    return fail(i, f'{p} is full')
                sup[x] = p
            else:
                return fail(i, f'unknown action {a}')
            placed.append(sup[x])
            held = None
        else:
            return fail(i, f'unknown action {a}')
    if held:
        return fail(len(plan) - 1, f'the plan ends holding {held}')
    return {'ok': True, 'error': None, 'support': sup, 'picks': picks, 'placed': placed}


def satisfies(scene: dict, support: dict, reference) -> bool:
    """Does the final arrangement make the reference formula true?"""
    sc = _sc(scene)
    return goal_logic.compile_goal(reference, sc).holds(goal_logic.View(support, sc['items']))


def base_of(support: dict, x: str):
    while support.get(x) in support:
        x = support[x]
    return support.get(x)


def check_plan(scene: dict, response: dict, chk: dict) -> list[str]:
    """The "_check" keys of examples/requests/*.json, judged on a replay. -> list of failures."""
    fails = []
    if chk.get('decision') and response.get('decision') != chk['decision']:
        fails.append(f"decision {response.get('decision')} != {chk['decision']}")
    if 'constraints' in chk and response.get('status') == 'ok':
        got = dict(response.get('constraints') or {})
        if got != chk['constraints']:
            fails.append(f"constraints {got} != {chk['constraints']}")
    if response.get('status') == 'ok' and (response.get('plan') or 'on' in chk or 'in' in chk):
        rp = replay(scene, response.get('plan') or [])
        if not rp['ok']:
            return fails + [rp['error']]
        sup = rp['support']
        for a, b in chk.get('on', []):
            if sup.get(a) != b:
                fails.append(f'{a} ends on {sup.get(a)}, not {b}')
        for a, z in chk.get('in', []):
            if base_of(sup, a) != z:
                fails.append(f'{a} ends in {base_of(sup, a)}, not {z}')
        for d in chk.get('no_place_on', []):
            if d in rp['placed']:
                fails.append(f'something was placed on {d}')
        for x in chk.get('never_pick', []):
            if x in rp['picks']:
                fails.append(f'{x} was moved')
        if chk.get('first_pick') and (not rp['picks'] or rp['picks'][0] != chk['first_pick']):
            fails.append(f"first pick {rp['picks'][0] if rp['picks'] else None} != {chk['first_pick']}")
        if chk.get('max_moves') is not None and len(rp['picks']) > chk['max_moves']:
            fails.append(f"{len(rp['picks'])} moves, the shortest plan has {chk['max_moves']}")
    return fails


# --------------------------------------------------------------------------- perception
def _key(d: dict) -> tuple:
    return (d.get('color'), d.get('shape'))


def _place_key(pid: str, d: dict):
    return openworld.colour_of(d.get('label'), pid) or pid


def perception_score(pred_scene: dict, truth_scene: dict) -> dict:
    """How well does the scene read from a photo match the true scene?

    Objects are matched by (colour, shape), which is what a camera can tell apart (the model
    chooses its own ids). recall = true objects found, precision = found objects that are real,
    support = share of true objects whose "rests on" is right (an object, a zone of the right
    colour, or the table). Also returns id maps to translate a plan onto the true scene.
    """
    t, p = _sc(truth_scene), _sc(pred_scene)
    pool = {}
    for tid, d in t['items'].items():
        pool.setdefault(_key(d), []).append(tid)
    item_map = {}
    for pid, d in p['items'].items():
        k = _key(d)
        if pool.get(k):
            item_map[pid] = pool[k].pop(0)
    t_places = {_place_key(q, d): q for q, d in t['env']['places'].items()}
    place_map = {q: t_places[_place_key(q, d)] for q, d in p['env']['places'].items()
                 if _place_key(q, d) in t_places}

    def target(sc, sup, ids, places):
        if sup == 'table':
            return 'table'
        if sup in sc['items']:
            return ('item', ids.get(sup))
        return ('place', places.get(sup))

    right = 0
    inverse = {v: k for k, v in item_map.items()}
    for tid, tsup in t['support'].items():
        pid = inverse.get(tid)
        if pid is None or pid not in p['support']:
            continue
        want = target(t, tsup, {i: i for i in t['items']}, {q: q for q in t['env']['places']})
        got = target(p, p['support'][pid], item_map, place_map)
        right += want == got
    n_t, n_p, m = len(t['items']), len(p['items']), len(item_map)
    out = {'n_truth': n_t, 'n_pred': n_p, 'matched': m,
           'recall': m / n_t if n_t else 1.0, 'precision': m / n_p if n_p else 0.0,
           'support_acc': right / n_t if n_t else 1.0,
           'zones_truth': len(t['env']['places']), 'zones_found': len(place_map)}
    out['exact'] = out['recall'] == 1 and out['precision'] == 1 and out['support_acc'] == 1
    return {**out, 'item_map': item_map, 'place_map': place_map}


def _rename(a: str, item_map: dict, place_map: dict) -> str:
    if a in item_map:
        return item_map[a]
    if a in place_map:
        return place_map[a]
    m = re.fullmatch(r'(.+)_s(\d+)', a)                     # a slot of a place: red_zone_s2
    if m and m.group(1) in place_map:
        return f'{place_map[m.group(1)]}_s{m.group(2)}'
    return a


def translate_plan(plan: list[dict], item_map: dict, place_map: dict) -> list[dict]:
    """Rename the ids of a plan made for a photo-derived scene to the true scene's ids."""
    return [{**s, 'args': [_rename(a, item_map, place_map) for a in s.get('args') or []]} for s in plan or []]


# --------------------------------------------------------------------------- one verdict
def judge(truth: dict, response: dict, expect: list[str] | None = None, reference=None,
          check: dict | None = None, plan: list[dict] | None = None) -> dict:
    """-> {"outcome", "passed", "reasons"}.

    expect  statuses that count as right (default ["ok"]); a refusal can be the right answer
    plan    the plan to replay if it needs renaming first (photo route); default: response plan
    """
    want = expect or ['ok']
    status = response.get('status')
    reasons = []
    if status not in want:
        outcome = {'clarify': 'asked', 'need_observation': 'asked', 'no_plan': 'no plan',
                   'ok': 'unexpected ok'}.get(status, 'error')
        if response.get('message'):
            reasons.append(str(response['message'])[:200])
        return {'outcome': outcome, 'passed': False, 'reasons': reasons}
    if status != 'ok':
        return {'outcome': 'correct', 'passed': True, 'reasons': []}
    steps = response.get('plan') if plan is None else plan
    rp = replay(truth, steps or [])
    if not rp['ok']:
        return {'outcome': 'wrong plan', 'passed': False, 'reasons': [rp['error']]}
    if reference is not None:
        try:
            if not satisfies(truth, rp['support'], reference):
                reasons.append('the final arrangement does not satisfy the task')
        except (goal_logic.FormulaError, openworld.SceneError) as e:
            reasons.append(f'reference could not be evaluated: {e}')
    if check:
        reasons += check_plan(truth, {**response, 'plan': steps}, check)
    return {'outcome': 'wrong plan' if reasons else 'correct', 'passed': not reasons, 'reasons': reasons}
