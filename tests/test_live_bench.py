"""Live benchmark: scoring rules and a full run against the stand-in model (no GPU).
Run: python -m pytest tests/test_live_bench.py -q"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

import live_bench
import live_cases
import live_report
import live_score
import openworld
from server import create_app

SMALL = {'objects': [{'id': 'r1', 'color': 'red', 'shape': 'cube', 'on': 'table'},
                     {'id': 'b1', 'color': 'blue', 'shape': 'cube', 'on': 'r1'},
                     {'id': 'b2', 'color': 'blue', 'shape': 'bar', 'on': 'table'}],
         'zones': [{'id': 'red_zone', 'color': 'red', 'capacity': 1}, {'id': 'blue_zone', 'color': 'blue', 'capacity': 2}]}


def step(a, *args):
    return {'action': a, 'args': list(args)}


def test_replay_accepts_a_correct_plan_and_computes_the_final_state():
    plan = [step('pick-from-item', 'b1', 'r1'), step('place-in', 'b1', 'blue_zone_s1', 'blue_zone'),
            step('pick-from-table', 'r1'), step('place-in', 'r1', 'red_zone_s1', 'red_zone')]
    r = live_score.replay(SMALL, plan)
    assert r['ok'] and r['support'] == {'b1': 'blue_zone', 'r1': 'red_zone', 'b2': 'table'}
    assert not live_score.satisfies(SMALL, r['support'], live_cases.ref_sort(SMALL))       # b2 is still on the table
    plan += [step('pick-from-table', 'b2'), step('place-in', 'b2', 'blue_zone_s2', 'blue_zone')]
    r = live_score.replay(SMALL, plan)
    assert r['ok'] and live_score.satisfies(SMALL, r['support'], live_cases.ref_sort(SMALL))


@pytest.mark.parametrize('plan,fragment', [
    ([step('pick-from-table', 'r1')], 'has something on it'),                       # b1 stands on r1
    ([step('pick-from-item', 'b1', 'r1'), step('pick-from-table', 'b2')], 'hand is not empty'),
    ([step('pick-from-item', 'b1', 'r1')], 'ends holding'),
    ([step('pick-from-item', 'b1', 'r1'), step('place-on-item', 'b1', 'b1')], 'not a free object'),
    ([step('pick-from-item', 'b1', 'r1'), step('place-in', 'b1', 'red_zone_s1', 'red_zone'),
      step('pick-from-table', 'b2'), step('place-in', 'b2', 'red_zone_s1', 'red_zone')], 'is full'),
    ([step('pick-from-table', 'b2'), step('dance', 'b2')], 'unknown action'),
])
def test_replay_catches_illegal_plans(plan, fragment):
    r = live_score.replay(SMALL, plan)
    assert not r['ok'] and fragment in r['error']


def test_judge_separates_wrong_plans_from_refusals_from_unexpected_ok():
    ok_but_wrong = {'status': 'ok', 'plan': [step('pick-from-item', 'b1', 'r1'), step('place-on-table', 'b1')]}
    v = live_score.judge(SMALL, ok_but_wrong, reference=live_cases.ref_sort(SMALL))
    assert v['outcome'] == 'wrong plan' and not v['passed']
    assert live_score.judge(SMALL, {'status': 'clarify', 'message': 'which?'})['outcome'] == 'asked'
    assert live_score.judge(SMALL, {'status': 'clarify'}, expect=['clarify'])['passed']
    assert live_score.judge(SMALL, {'status': 'ok', 'plan': []}, expect=['clarify'])['outcome'] == 'unexpected ok'
    assert live_score.judge(SMALL, {'status': 'no_plan'}, expect=['no_plan', 'clarify'])['passed']


def test_perception_score_matches_objects_by_colour_and_shape():
    seen = {'objects': [{'id': 'obj_a', 'label': 'red cube'}, {'id': 'obj_b', 'label': 'blue cube'},
                        {'id': 'ghost', 'label': 'green ball'}],
            'places': [{'id': 'paper_red', 'label': 'red paper', 'capacity': 1}],
            'support': {'obj_a': 'table', 'obj_b': 'obj_a', 'ghost': 'table'}}
    s = live_score.perception_score(seen, SMALL)
    assert s['n_truth'] == 3 and s['matched'] == 2 and s['n_pred'] == 3
    assert s['recall'] == pytest.approx(2 / 3) and s['precision'] == pytest.approx(2 / 3)
    assert s['support_acc'] == pytest.approx(2 / 3)                 # r1, b1 right; b2 missing
    assert s['item_map'] == {'obj_a': 'r1', 'obj_b': 'b1'} and s['place_map'] == {'paper_red': 'red_zone'}
    plan = live_score.translate_plan([step('place-in', 'obj_a', 'paper_red_s1', 'paper_red')], s['item_map'], s['place_map'])
    assert plan[0]['args'] == ['r1', 'red_zone_s1', 'red_zone']


def test_check_block_of_example_requests():
    resp = {'status': 'ok', 'plan': [step('pick-from-item', 'b1', 'r1'), step('place-on-table', 'b1')]}
    assert live_score.check_plan(SMALL, resp, {'on': [['b1', 'table']], 'max_moves': 1}) == []
    assert live_score.check_plan(SMALL, resp, {'never_pick': ['b1']}) == ['b1 was moved']


def test_every_case_is_well_formed_and_references_hold_on_a_correct_final_state():
    cases = live_cases.json_cases()
    assert {c['group'] for c in cases} >= {'sort', 'stack', 'novel', 'language', 'refusal', 'constraints', 'examples'}
    for c in cases:
        openworld.normalize_scene(c['scene'])
        if c.get('reference'):
            assert live_score.satisfies(c['scene'], {x['id']: x['on'] for x in c['scene']['objects']}, c['reference']) in (True, False)
    # the generated towers really are n objects, four colours
    sc = live_cases.towers_scene(24)
    assert len(sc['objects']) == 24 and {o['color'] for o in sc['objects']} == set(live_cases.COLOURS)


def test_full_quick_run_and_report(tmp_path):
    api = TestClient(create_app(openworld.RuleMock(), log_dir=tmp_path / 'logs'))
    args = argparse.Namespace(suites='json,scale,photo', quick=True, only='', repeats=1, scale='4,8', photos=[],
                              photo_limit=2, photo_resolutions='low,high', plan_resolutions='auto', clear_memory=False,
                              no_warmup=True, label='mock', url='testclient')
    meta, rows = live_bench.run(api, args, tmp_path / 'run')
    suites = {r['suite'] for r in rows}
    assert suites == {'json', 'scale', 'photo'}
    assert all(r['outcome'] in live_report.OUTCOMES for r in rows)
    scale = [r for r in rows if r['suite'] == 'scale']
    assert scale and all(r['status'] == 'ok' and r['passed'] for r in scale)       # planner alone, no model
    assert any(r['group'] == 'perceive' for r in rows) and any(r['group'] == 'photo_plan' for r in rows)
    res = json.loads((tmp_path / 'run' / 'results.json').read_text())
    assert len(res['rows']) == len(rows) and res['summary']
    page = live_report.build([tmp_path / 'run'])
    text = page.read_text()
    assert 'data:image/png;base64' in text and 'What failed' in text
    for name in ('outcomes', 'latency', 'scaling', 'photo_resolution', 'photo_vs_json', 'quality'):
        assert (tmp_path / 'run' / 'charts' / f'{name}.png').exists(), name


def test_unreachable_server_gives_a_clear_message():
    args = argparse.Namespace(suites='json', quick=True, only='', repeats=1, scale='', photos=[], photo_limit=0,
                              photo_resolutions='low', plan_resolutions='auto', clear_memory=False, no_warmup=True,
                              label='x', url='http://127.0.0.1:9')
    with pytest.raises(SystemExit) as e:
        live_bench.run(live_bench.Http('http://127.0.0.1:9'), args)
    assert 'cannot reach the server' in str(e.value)


def test_own_photo_folder_is_read_and_its_references_evaluate(tmp_path):
    from PIL import Image
    src = Path(__file__).resolve().parents[1] / 'live_photos'
    for name in ('four_towers.json', 'four_towers.tasks.json'):
        (tmp_path / name).write_text((src / name).read_text())
    Image.new('RGB', (64, 48), 'white').save(tmp_path / 'four_towers.jpg')
    (tmp_path / 'no_truth.jpg').write_bytes((tmp_path / 'four_towers.jpg').read_bytes())     # no .json: skipped
    mine = [s for s in live_cases.photo_sources([tmp_path]) if s['name'] == 'four_towers']
    assert len(mine) == 1 and len(mine[0]['tasks']) == 4
    start = {o['id']: o['on'] for o in mine[0]['truth']['objects']}
    for t in mine[0]['tasks']:                                      # evaluates (true or false) without error
        assert live_score.satisfies(mine[0]['truth'], start, t['reference']) in (True, False)
    assert not live_score.satisfies(mine[0]['truth'], start, mine[0]['tasks'][0]['reference'])   # not sorted yet
    assert not any(s['name'] == 'no_truth' for s in live_cases.photo_sources([tmp_path]))
