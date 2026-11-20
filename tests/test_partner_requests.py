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
    assert r['status'] == 'no_plan' and 'without undoing goals 1-1' in r['message']
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
    assert set(s) == {'request_id', 'status', 'plan', 'quality'} and s['quality'] == {'moves': 16, 'optimal': True}
    assert s['plan'][0] == ' '.join([full.json()['plan'][0]['action'], *full.json()['plan'][0]['args']])
    assert len(small.content) < len(full.content) / 3
    r = client.post('/replan', json={'request_id': s['request_id'], 'scene': FOUR, 'compact': True}).json()
    assert r['status'] == 'ok' and r['decision'] == 'kept' and set(r) <= {'request_id', 'status', 'plan', 'quality', 'decision'}
    q = client.post('/plan', json={'instruction': 'put it on the other one', 'scene': FOUR, 'compact': True}).json()
    assert q['status'] == 'clarify' and q['message']
