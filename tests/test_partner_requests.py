"""The robot team's requests of 20 Nov 2026: ordered goals, constraints in /solve, towers in the zones,
size and height limits, failed steps in /replan, compact answers. Run: python -m pytest tests/test_partner_requests.py -q"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

import openworld
from server import create_app

ROOT = Path(__file__).resolve().parents[1]
FOUR = json.loads((ROOT / 'examples' / 'requests' / '15_four_towers_sort.json').read_text())['scene']
ABC = {'objects': [{'id': 'a', 'on': 'table'}, {'id': 'b', 'on': 'table'}, {'id': 'c', 'on': 'table'}]}


@pytest.fixture(scope='module')
def client(tmp_path_factory):
    return TestClient(create_app(openworld.RuleMock(), log_dir=tmp_path_factory.mktemp('logs')))


def final(scene, plan):
    return openworld._apply_steps({o['id']: o['on'] for o in scene['objects']}, plan)


# 1 ------------------------------------------------------------------- goal order
def test_ordered_goals_are_met_in_order_or_refused(client):
    later_first = [{'object': 'c', 'on': 'b'}, {'object': 'b', 'on': 'a'}]       # c on b first, then b cannot move
    r = client.post('/solve', json={'scene': ABC, 'goal': later_first}).json()
    assert r['status'] == 'ok'                                                    # unordered: fine (b on a, then c on b)
    r = client.post('/solve', json={'scene': ABC, 'goal': later_first, 'ordered': True}).json()
    assert r['status'] == 'no_plan' and 'cannot be met in this order' in r['message']
    r = client.post('/solve', json={'scene': ABC, 'goal': later_first[::-1], 'ordered': True}).json()
    assert r['status'] == 'ok' and r['order'] == {'achieved_at_move': [1, 2], 'in_order': True}
    assert [s['pddl'] for s in r['plan']] == ['(pick-from-table b)', '(place-on-item b a)',
                                              '(pick-from-table c)', '(place-on-item c b)']


def test_the_model_can_mark_an_instruction_as_ordered(client):
    r = client.post('/plan', json={'instruction': 'first put b on a, then put c on b', 'scene': ABC}).json()
    assert r['status'] == 'ok' and r['ordered'] is True and r['order']['in_order']


# 2 ------------------------------------------------------------------- constraints in /solve
def test_solve_applies_constraints(client):
    r = client.post('/solve', json={'scene': FOUR, 'goal': [{'sort_by_color': True}],
                                    'constraints': {'no_table': True}}).json()
    assert r['status'] == 'ok' and r['constraints'] == {'no_table': True}
    assert not any(s['action'] == 'place-on-table' for s in r['plan'])
    r = client.post('/solve', json={'scene': FOUR, 'goal': [{'sort_by_color': True}],
                                    'constraints': {'keep': ['red_block_1']}}).json()
    assert r['status'] == 'no_plan'                                    # red_block_1 must go to its zone


# 3 ------------------------------------------------------------------- towers in the zones
@pytest.mark.parametrize('capacity', [1, 4])
def test_one_tower_per_colour_in_its_zone_without_the_table(client, capacity):
    sc = copy.deepcopy(FOUR)
    for z in sc['zones']:
        z['capacity'] = capacity
    r = client.post('/solve', json={'scene': sc, 'goal': [{'stack_in_zones': True}],
                                    'constraints': {'no_table': True}}).json()
    assert r['status'] == 'ok' and r['quality']['optimal'] and r['quality']['moves'] == 16
    assert r['goal'] == [{'stack_in_zones': True}] and r['timings']['plan_s'] < 8
    sup, col = final(sc, r['plan']), {o['id']: o['color'] for o in sc['objects']}
    for x, y in sup.items():                                           # own colour below, or its own zone
        assert (y in col and col[y] == col[x]) or y == f'{col[x]}_zone'
    assert sum(1 for x, y in sup.items() if y.endswith('_zone')) == 4  # one base per zone
    assert not any(s['action'] == 'place-on-table' for s in r['plan'])


def test_zone_towers_from_the_instruction(client):
    r = client.post('/plan', json={'instruction': 'build one tower per colour in the zones', 'scene': FOUR}).json()
    assert r['status'] == 'ok' and r['goal'] == [{'stack_in_zones': True}]


# 4 ------------------------------------------------------------------- size and height limits
def limited(zone_mm=None, zone_objects=None, rules=None):
    sc = copy.deepcopy(FOUR)
    size = {'bar': 80, 'block': 60, 'cube': 40, 'cylinder': 40}
    for o in sc['objects']:
        o['size_mm'], o['height_mm'] = size[o['shape']], 30
    for z in sc['zones']:
        if zone_mm:
            z['max_height_mm'] = zone_mm
        if zone_objects:
            z['max_top'] = zone_objects                                # the name the robot team already sends
    if rules:
        sc['rules'] = rules
    return sc


def test_larger_never_on_smaller_and_zone_height_respected(client):
    sc = limited(zone_mm=130)
    r = client.post('/solve', json={'scene': sc, 'goal': [{'stack_in_zones': True}]}).json()
    assert r['status'] == 'ok' and r['verified']
    sup, size = final(sc, r['plan']), {o['id']: o['size_mm'] for o in sc['objects']}
    assert all(size[x] <= size[y] for x, y in sup.items() if y in size)
    tower = {}
    for x in sup:
        n, cur = 1, x
        while sup[cur] in size:
            cur, n = sup[cur], n + 1
        tower[x] = n
    assert max(tower.values()) * 30 <= 130


def test_impossible_limits_are_explained_at_once(client):
    for sc, word in ((limited(zone_mm=90), '120 mm tower'), (limited(zone_objects=3), 'at most 3 allowed')):
        r = client.post('/solve', json={'scene': sc, 'goal': [{'stack_in_zones': True}]}).json()
        assert r['status'] == 'no_plan' and word in r['message'] and r['timings']['plan_s'] < 0.5


def test_limits_also_apply_to_fixed_goals_and_the_table(client):
    sc = limited(rules={'max_objects': 2})
    goal = [{'object': 'red_block_1', 'on': 'green_cube_2'}, {'object': 'blue_cube_1', 'on': 'red_block_1'}]
    r = client.post('/solve', json={'scene': sc, 'goal': goal}).json()
    assert r['status'] == 'no_plan'                                    # green_cube_2 is on a tower already


# 5 ------------------------------------------------------------------- failed step
def test_a_failed_step_is_never_proposed_again(client):
    sc = {'objects': [{'id': 'r1', 'color': 'red', 'on': 'table'}, {'id': 'b1', 'color': 'blue', 'on': 'r1'},
                      {'id': 'g1', 'color': 'green', 'on': 'table'}],
          'zones': [{'id': 'red_zone', 'color': 'red', 'capacity': 1}]}
    r = client.post('/solve', json={'scene': sc, 'goal': [{'object': 'r1', 'in': 'red_zone'}]}).json()
    assert r['status'] == 'ok' and r['plan'][1]['pddl'] == '(place-on-table b1)'
    for failed in (['(place-on-table b1)'], [{'action': 'place-on-table', 'args': ['b1']}], [{'step': 2}]):
        r2 = client.post('/replan', json={'request_id': r['request_id'], 'scene': sc, 'failed': failed}).json()
        assert r2['status'] == 'ok' and r2['decision'] == 'replanned'
        assert '(place-on-table b1)' not in [s['pddl'] for s in r2['plan']]
        assert ['b1', 'table'] in [list(t) for t in r2['constraints']['never_on']]
    r3 = client.post('/replan', json={'request_id': r['request_id'], 'scene': sc,
                                      'failed': ['(pick-from-item b1 r1)']}).json()
    assert r3['status'] == 'no_plan' and 'b1 will not be picked again' in r3['message']


def test_failures_add_up_over_replans(client):
    sc = {'objects': [{'id': 'r1', 'color': 'red', 'on': 'table'}, {'id': 'b1', 'color': 'blue', 'on': 'r1'},
                      {'id': 'g1', 'color': 'green', 'on': 'table'}, {'id': 'y1', 'color': 'yellow', 'on': 'table'}],
          'zones': [{'id': 'red_zone', 'color': 'red', 'capacity': 1}]}
    r = client.post('/solve', json={'scene': sc, 'goal': [{'object': 'r1', 'in': 'red_zone'}]}).json()
    r2 = client.post('/replan', json={'request_id': r['request_id'], 'scene': sc, 'failed': ['(place-on-table b1)']}).json()
    target = r2['plan'][1]['args'][-1]
    r3 = client.post('/replan', json={'request_id': r2['request_id'], 'scene': sc,
                                      'failed': [r2['plan'][1]['pddl']]}).json()
    assert r3['status'] == 'ok' and r3['plan'][1]['args'][-1] not in ('table', target)


# 6 ------------------------------------------------------------------- compact answers
def test_compact_answer_is_small_and_replan_still_works(client):
    body = {'scene': FOUR, 'goal': [{'sort_by_color': True}]}
    full = client.post('/solve', json=body)
    small = client.post('/solve', json={**body, 'compact': True})
    s = small.json()
    assert set(s) == {'request_id', 'status', 'plan', 'quality', 'total_s'} and s['quality'] == {'moves': 16, 'optimal': True}
    assert s['plan'][0] == ' '.join([full.json()['plan'][0]['action'], *full.json()['plan'][0]['args']])
    assert len(small.content) < len(full.content) / 3
    r = client.post('/replan', json={'request_id': s['request_id'], 'scene': FOUR, 'compact': True}).json()
    assert r['status'] == 'ok' and r['decision'] == 'kept' and set(r) <= {'request_id', 'status', 'plan', 'quality', 'decision', 'total_s'}
    q = client.post('/plan', json={'instruction': 'put it on the other one', 'scene': FOUR, 'compact': True}).json()
    assert q['status'] == 'clarify' and q['message']


# ------------------------------------------------------------------- round 2 (v2 feedback)
import collections
import gzip
import random

import formula_plan


def brute_force_ordered(sc, goal, cons=None):
    """Shortest plan length under the same order rules, by plain breadth-first search (no estimate)."""
    mover, n = formula_plan.Mover(sc, cons or {}), len(goal)
    fact = formula_plan._fact

    def advance(sup, k):
        while k < n and fact(goal[k], sup):
            k += 1
        return k
    start = dict(sc['support'])
    q = collections.deque([(start, advance(start, 0), 0)])
    seen = {(tuple(sorted(start.items())), q[0][1])}
    while q:
        sup, k, d = q.popleft()
        if k == n:
            return d
        for x, dst in mover.moves(sup):
            new = {**sup, x: dst}
            if any(not fact(goal[j], new) for j in range(k)) or \
                    any(fact(goal[j], new) and not fact(goal[j], sup) for j in range(k + 1, n)):
                continue
            k2 = advance(new, k)
            key = (tuple(sorted(new.items())), k2)
            if key not in seen:
                seen.add(key)
                q.append((new, k2, d + 1))
    return None


@pytest.mark.parametrize('seed', range(12))
def test_ordered_search_finds_the_true_shortest_plan(seed):
    rnd = random.Random(seed)
    ids = [f'o{i}' for i in range(6)]
    objs, tops = [], []
    for x in ids:                                                   # random towers
        on = rnd.choice(tops + ['table', 'table'])
        if on != 'table':
            tops.remove(on)
        objs.append({'id': x, 'on': on})
        tops.append(x)
    scene = {'objects': objs, 'zones': [{'id': 'zone_a', 'capacity': 2}, {'id': 'zone_b', 'capacity': 2}]}
    sc = openworld.normalize_scene(scene)
    a, b, c, d = rnd.sample(ids, 4)
    goal = [('in', a, 'zone_a'), ('on', b, a), ('on-table', c), ('in', d, 'zone_b')]
    best = brute_force_ordered(sc, goal)
    r = openworld.plan_for(sc, goal, ordered=True)
    if best is None:
        assert r['status'] == 'no_plan'
    else:
        assert r['status'] == 'ok' and r['quality']['moves'] == best and r['quality']['optimal'], (seed, best, r['quality'])
        assert r['order']['in_order']


def test_ordered_answers_do_not_change_between_runs(client):
    goal = [{'object': 'green_cube_2', 'in': 'green_zone'}, {'object': 'red_cube_1', 'in': 'red_zone'},
            {'object': 'yellow_bar_2', 'on': 'red_cube_1'}, {'object': 'blue_cube_1', 'on': 'table'}]
    plans = [client.post('/solve', json={'scene': FOUR, 'goal': goal, 'ordered': True}).json() for _ in range(3)]
    assert all(p['status'] == 'ok' for p in plans)
    assert len({json.dumps(p['plan']) for p in plans}) == 1 and plans[0]['quality']['optimal']


def test_several_formula_entries_must_all_hold(client):
    goal = [{'formula': {'in': ['red_cube_1', 'red_zone']}}, {'formula': {'in': ['blue_cube_1', 'blue_zone']}},
            {'object': 'green_cube_2', 'in': 'green_zone'}]
    r = client.post('/solve', json={'scene': FOUR, 'goal': goal}).json()
    assert r['status'] == 'ok'
    sup = final(FOUR, r['plan'])
    assert sup['red_cube_1'] == 'red_zone' and sup['blue_cube_1'] == 'blue_zone' and sup['green_cube_2'] == 'green_zone'


def test_quality_lists_the_objects_the_lower_bound_counts(client):
    r = client.post('/solve', json={'scene': FOUR, 'goal': [{'stack_in_zones': True}]}).json()
    assert r['quality']['lower_bound'] == len(r['quality']['must_move']) == 16
    r = client.post('/solve', json={'scene': ABC, 'goal': [{'object': 'c', 'on': 'b'}]}).json()
    assert r['quality']['must_move'] == ['c']


def test_server_time_and_gzip_both_ways(client):
    body = json.dumps({'scene': FOUR, 'goal': [{'sort_by_color': True}], 'compact': True}).encode()
    r = client.post('/solve', content=gzip.compress(body),
                    headers={'Content-Type': 'application/json', 'Content-Encoding': 'gzip', 'Accept-Encoding': 'gzip'})
    assert r.status_code == 200 and r.headers['content-encoding'] == 'gzip'
    assert float(r.headers['x-server-ms']) > 0 and r.json()['total_s'] > 0
    assert int(r.headers['content-length']) < len(r.content)            # compressed on the wire
    bad = client.post('/solve', content=b'nope', headers={'Content-Type': 'application/json', 'Content-Encoding': 'gzip'})
    assert bad.status_code == 400


# ------------------------------------------------------------------- round 3 (farby: exact search, /check)
def zone_scene(seed, n_per_colour=2, colours=('red', 'blue')):
    rnd = random.Random(seed)
    objs = [{'id': f'{c}_plank', 'color': c, 'size_mm': 80, 'height_mm': 20, 'on': f'{c}_zone'} for c in colours]
    rest = [{'id': f'{c}_{k}', 'color': c, 'size_mm': rnd.choice([25, 30, 40]), 'height_mm': 30}
            for c in colours for k in range(n_per_colour)]
    rnd.shuffle(rest)
    tops = {}
    for k, o in enumerate(rest):
        o['on'] = tops.get(k % 2, 'table')
        tops[k % 2] = o['id']
    zones = [{'id': f'{c}_zone', 'color': c, 'capacity': 1, 'max_height_mm': 110} for c in colours]
    return {'objects': objs + rest, 'zones': zones}


def test_explain_agrees_with_the_search_rules():
    for seed in range(15):
        scene = zone_scene(seed, 3)
        sc = openworld.normalize_scene(scene)
        cons = {'keep': ['red_plank'], 'never_on': [['blue_0', 'red_1']]} if seed % 2 else {'no_table': True}
        mover, sup = formula_plan.Mover(sc, cons), dict(sc['support'])
        rnd = random.Random(seed)
        for _ in range(6):                                   # several arrangements along random moves
            allowed = set(mover.moves(sup))
            for x in sc['items']:
                for dst in ['table', *sc['items'], *sc['env']['places']]:
                    if x in sup and dst != sup.get(x):
                        assert ((x, dst) in allowed) == (formula_plan.explain(mover, sup, x, dst) is None), (seed, x, dst)
            if not allowed:
                break
            x, dst = rnd.choice(sorted(allowed))
            sup[x] = dst


@pytest.mark.parametrize('seed', range(8))
def test_zone_towers_exact_search_is_truly_shortest(seed):
    scene = zone_scene(seed)
    sc = openworld.normalize_scene(scene)
    cons = {'no_table': True, 'keep': ['red_plank', 'blue_plank']} if seed % 2 else {}
    zone_of = {'red': 'red_zone', 'blue': 'blue_zone'}
    import goal_logic
    goal = goal_logic.compile_goal(goal_logic.stack_in_zones_formula(zone_of), sc)
    mover = formula_plan.Mover(sc, cons)
    start = dict(sc['support'])
    q, seen, best = collections.deque([(start, 0)]), {tuple(sorted(start.items()))}, None
    while q:                                                 # plain breadth-first: the true shortest
        sup, d = q.popleft()
        if goal.holds(goal_logic.View(sup, sc['items'])):
            best = d
            break
        for x, dst in mover.moves(sup):
            new = {**sup, x: dst}
            k = tuple(sorted(new.items()))
            if k not in seen:
                seen.add(k)
                q.append((new, d + 1))
    r = openworld.run(scene=scene, goal=[{'stack_in_zones': True}], constraints=cons)
    if best is None:
        assert r['status'] == 'no_plan'
    else:
        assert r['status'] == 'ok' and r['quality']['moves'] == best and r['quality']['optimal'], (seed, best, r['quality'])


def test_check_accepts_our_plans_and_names_the_rejected_step(client):
    cons = {'no_table': True, 'keep': ['red_plank', 'blue_plank']}
    goal = [{'stack_in_zones': True}]
    for seed in range(20):                                   # the first solvable random scene
        scene = zone_scene(seed, 3)
        answer = client.post('/solve', json={'scene': scene, 'goal': goal, 'constraints': cons, 'compact': True}).json()
        if answer['status'] == 'ok' and answer['plan']:
            break
    plan = answer['plan']
    ok = client.post('/check', json={'scene': scene, 'goal': goal, 'constraints': cons, 'plan': plan}).json()
    assert ok['status'] == 'ok' and ok['goal_reached'] and ok['moves'] == len(plan) // 2
    sup = {o['id']: o['on'] for o in scene['objects']}
    top = next(x for x in sup if x not in sup.values() and not sup[x].endswith('_zone'))
    bad = [f'pick-from-{"table" if sup[top] == "table" else "item"} {top}' + ('' if sup[top] == 'table' else f' {sup[top]}'),
           f'place-on-table {top}']
    r = client.post('/check', json={'scene': scene, 'goal': goal, 'constraints': cons, 'plan': bad}).json()
    assert r['status'] == 'rejected' and r['step'] == 2 and 'table may not be used' in r['reason']
    r = client.post('/check', json={'scene': scene, 'goal': goal, 'constraints': cons,
                                    'plan': ['pick-from-place red_plank red_zone_s1 red_zone']}).json()
    assert r['status'] == 'rejected' and r['step'] == 1 and ('keep' in r['reason'] or 'has something on it' in r['reason'])
    r = client.post('/check', json={'scene': scene, 'goal': goal, 'constraints': cons, 'plan': []}).json()
    assert r['status'] == 'rejected' and 'goal is not reached' in r['reason']
