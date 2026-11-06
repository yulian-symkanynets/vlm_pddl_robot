"""Server + open-world pipeline with the rule-based mock (no GPU). Run: python -m pytest tests -q"""
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import yaml
from fastapi.testclient import TestClient

import openworld
from server import create_app

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / 'examples'
CASES = [c for c in yaml.safe_load((EX / 'cases.yaml').read_text())['cases'] if not c.get('real_model_only')]


@pytest.fixture(scope='module')
def client(tmp_path_factory):
    return TestClient(create_app(openworld.RuleMock(), log_dir=tmp_path_factory.mktemp('logs')))


def scene(name):
    return json.loads((EX / 'scenes' / name).read_text())


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['name'])
def test_example_cases(client, case):
    sc = json.loads((EX / case['scene']).read_text()) if case.get('scene') else None
    if case.get('goal'):
        r = client.post('/solve', json={'scene': sc, 'goal': case['goal']})
    elif sc is not None:
        r = client.post('/plan', json={'instruction': case['instruction'], 'scene': sc})
    else:
        files = [('files', (Path(p).name, (EX / p).read_bytes(), 'image/jpeg')) for p in case['images']]
        r = client.post('/plan/upload', data={'instruction': case['instruction']}, files=files)
    assert r.status_code == 200
    res = r.json()
    assert res['status'] == case['expect'], res.get('message')
    if res['status'] == 'ok':
        assert res['verified'] and res['plan']


def test_health(client):
    assert client.get('/health').json()['model'] == 'mock'


def test_box_plan_opens_lid_and_unstacks_first(client):
    res = client.post('/plan', json={'instruction': 'put the yellow block into the box',
                                     'scene': scene('box.json')}).json()
    acts = [s['action'] for s in res['plan']]
    assert acts[0] == 'open-lid' or acts.index('open-lid') < acts.index('place-in')
    assert acts.index('pick-from-item') < acts.index('pick-from-table')   # red cylinder off first


def test_too_tall_is_explained(client):
    res = client.post('/solve', json={'scene': scene('shelf.json'),
                                      'goal': [{'object': 'blue_cylinder', 'in': 'low_shelf'}]}).json()
    assert res['status'] == 'no_plan' and 'too tall' in res['message']


def test_semantic_path_sends_few_bytes(client):
    res = client.post('/plan', json={'instruction': 'put the green block on the red cylinder',
                                     'scene': scene('stacks.json')}).json()
    img = (ROOT / 'images' / 'scene5_reorder_close.jpg').stat().st_size
    assert res['status'] == 'ok' and res['received_bytes'] < img / 50


def test_base64_images(client):
    b = base64.b64encode((ROOT / 'images' / 'scene5_reorder_close.jpg').read_bytes()).decode()
    res = client.post('/plan', json={'instruction': 'put the green block on the red cylinder',
                                     'images_b64': [b]}).json()
    assert res['status'] == 'ok' and 'perceive' in res['model_answers']


def test_request_is_logged(client):
    res = client.post('/plan', json={'instruction': 'put the yellow block in the box',
                                     'scene': scene('box.json')}).json()
    got = client.get(f'/requests/{res["request_id"]}').json()
    assert got['response']['status'] == 'ok'


def test_perceive_endpoint(client):
    files = [('files', ('a.jpg', (ROOT / 'images' / 'scene5_reorder_close.jpg').read_bytes(), 'image/jpeg'))]
    res = client.post('/perceive', files=files).json()
    assert res['status'] == 'ok' and len(res['scene']['objects']) == 4


# ---- open-world details (no HTTP)
def test_labels_and_free_names_are_grounded():
    sc = openworld.normalize_scene({
        'objects': [{'id': 'Red Cylinder'}, {'id': 'obj 2', 'label': 'yellow block'}],
        'places': [{'id': 'the box', 'kind': 'box', 'open': False}, {'id': 'table'}],
        'support': {'Red Cylinder': 'yellow block', 'obj 2': 'table'}})
    assert set(sc['items']) == {'red_cylinder', 'obj_2'} and set(sc['env']['places']) == {'the_box'}
    goal, err = openworld.resolve_goal([{'object': 'yellow block', 'on': 'the box'}], sc)
    assert err is None and goal == [('in', 'obj_2', 'the_box')]       # "on a place" -> in


def test_impossible_goals_are_rejected_before_planning():
    sc = openworld.normalize_scene(scene('stacks.json'))
    for g, msg in [([{'object': 'red_cylinder', 'on': 'red_cylinder'}], 'itself'),
                   ([{'object': 'red_cylinder', 'on': 'green_block'},
                     {'object': 'green_block', 'on': 'red_cylinder'}], 'circular'),
                   ([{'object': 'red_zone', 'on': 'table'}], 'place')]:
        _, err = openworld.resolve_goal(g, sc)
        assert msg in err


def test_model_question_becomes_clarify():
    class Asks:
        key = 'asks'
        def generate(self, *a, **kw):
            return {'text': '{"goal": [], "question": "Which block?"}', 'generate_s': 0}
    res = openworld.run('put the block away', scene=scene('stacks.json'), vlm=Asks())
    assert res['status'] == 'clarify' and res['message'] == 'Which block?'


def test_web_page_and_examples(client):
    r = client.get('/')
    assert r.status_code == 200 and 'Robot Task Planner' in r.text
    ex = client.get('/examples').json()
    assert any(e.get('images') for e in ex) and any(e.get('scene') for e in ex)
    assert client.get(ex[-1]['images'][0] if ex[-1].get('images') else '/images/scene5_reorder_close.jpg').status_code == 200


def test_unusable_model_answer_is_retried():
    class FlakyThenGood(openworld.RuleMock):
        calls = 0
        def generate(self, image_paths, prompt, max_visual_tokens=1024, max_new_tokens=None,
                     sample=False, context=None):
            FlakyThenGood.calls += 1
            if context['mode'] == 'ow_goal' and not sample:
                return {'text': 'Sure! The goal is blue on yellow.', 'generate_s': 0}   # no JSON
            return super().generate(image_paths, prompt, max_visual_tokens, max_new_tokens, sample, context)
    res = openworld.run_until_ok(3, instruction='put the green block on the red cylinder',
                                 scene=scene('stacks.json'), vlm=FlakyThenGood())
    assert res['status'] == 'ok' and [a['status'] for a in res['attempts']] == ['error', 'ok']


def test_real_answers_are_not_retried():
    res = openworld.run_until_ok(3, instruction='put the purple ball in the red zone',
                                 scene=scene('stacks.json'), vlm=openworld.RuleMock())
    assert res['status'] == 'clarify' and len(res['attempts']) == 1


def test_mock_understands_both_parts_of_and():
    res = openworld.run('put the blue cylinder on the yellow block and the green block on the red cylinder',
                        scene=scene('stacks.json'), vlm=openworld.RuleMock())
    assert res['status'] == 'ok' and len(res['goal']) == 2 and len(res['plan']) == 6
