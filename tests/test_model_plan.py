"""POST /plan/model: the model writes the moves, a gate lets only safe moves through, the planner finishes.
Scripted models make the mistakes real models make. Run: python -m pytest tests/test_model_plan.py -q"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

import live_cases
import openworld
from server import create_app

SCENE = {'objects': [{'id': 'r1', 'color': 'red', 'on': 'table'}, {'id': 'b1', 'color': 'blue', 'on': 'r1'},
                     {'id': 'g1', 'color': 'green', 'on': 'table'}],
         'zones': [{'id': 'red_zone', 'color': 'red', 'capacity': 1}, {'id': 'blue_zone', 'color': 'blue', 'capacity': 1}]}
GOAL = [{'object': 'r1', 'in': 'red_zone'}, {'object': 'b1', 'in': 'blue_zone'}]     # shortest: 2 moves


class Scripted:
    """Answers the move prompts from a list (one answer per call) and records the prompts."""
    key = 'scripted'

    def __init__(self, *answers):
        self.answers, self.prompts = list(answers), []

    def generate(self, image_paths, prompt, max_visual_tokens=1024, max_new_tokens=None, sample=False, context=None):
        self.prompts.append(prompt)
        a = self.answers[min(len(self.prompts) - 1, len(self.answers) - 1)]
        return {'text': a if isinstance(a, str) else json.dumps({'moves': a}), 'generate_s': 0.01}


def ask(model, tmp_path, **body):
    c = TestClient(create_app(model, log_dir=tmp_path))
    return c.post('/plan/model', json={'scene': SCENE, 'goal': GOAL, 'repairs': 0, **body}).json()


def test_a_wrong_first_move_never_reaches_the_robot(tmp_path):
    r = ask(Scripted([['r1', 'red_zone'], ['b1', 'blue_zone']]), tmp_path)        # r1 has b1 on it
    assert r['status'] == 'ok' and r['verified'] and r['quality']['moves'] == 2
    assert r['quality']['model_moves'] == 0 and r['quality']['planner_moves'] == 2
    assert 'b1 is on it' in r['model_plan']['rejected'][0]['reason']


def test_right_moves_are_kept_and_the_planner_finishes(tmp_path):
    r = ask(Scripted([['b1', 'blue_zone'], ['r1', 'blue_zone']]), tmp_path)       # second move: zone full
    assert r['status'] == 'ok' and r['quality']['model_moves'] == 1 and r['quality']['planner_moves'] == 1
    assert 'full' in r['model_plan']['rejected'][0]['reason']
    assert [s['pddl'] for s in r['plan']][:2] == ['(pick-from-item b1 r1)', '(place-in b1 blue_zone_s1 blue_zone)']


def test_repair_tells_the_model_why_and_continues_from_there(tmp_path):
    m = Scripted([['b1', 'blue_zone'], ['r1', 'blue_zone']], [['r1', 'red_zone']])
    r = ask(m, tmp_path, repairs=1)
    assert r['status'] == 'ok' and r['quality']['model_moves'] == 2 and r['quality']['planner_moves'] == 0
    assert r['planner'] == 'model' and r['model_plan']['calls'] == 2 and r['quality']['optimal']
    assert 'was rejected' in m.prompts[1] and 'full' in m.prompts[1] and 'b1 -> blue_zone' in m.prompts[1]


def test_a_loop_is_stopped(tmp_path):
    r = ask(Scripted([['b1', 'table'], ['b1', 'r1'], ['b1', 'table']]), tmp_path)
    assert r['status'] == 'ok' and r['quality']['model_moves'] == 1
    assert 'loop' in r['model_plan']['rejected'][0]['reason']


def test_optimal_gate_rejects_legal_but_wasteful_moves(tmp_path):
    wasteful = [['b1', 'table'], ['r1', 'red_zone'], ['b1', 'blue_zone']]          # legal, 3 moves instead of 2
    legal = ask(Scripted(wasteful), tmp_path)
    assert legal['quality']['model_moves'] == 3 and legal['quality']['extra_moves'] == 1
    assert not legal['quality']['optimal']
    strict = ask(Scripted(wasteful), tmp_path, gate='optimal')
    assert strict['quality']['model_moves'] == 0 and strict['quality']['moves'] == 2 and strict['quality']['optimal']
    assert 'not on a shortest plan' in strict['model_plan']['rejected'][0]['reason']


def test_unreadable_answer_falls_back_to_the_planner(tmp_path):
    r = ask(Scripted('sorry, I cannot help with that'), tmp_path)
    assert r['status'] == 'ok' and r['quality']['model_moves'] == 0 and r['planner'].startswith('planner')
    assert 'JSON' in r['model_plan']['rejected'][0]['reason']


def test_rules_in_the_prompt_and_constraints_enforced(tmp_path):
    m = Scripted([['b1', 'table'], ['r1', 'red_zone'], ['b1', 'blue_zone']])
    r = ask(m, tmp_path, constraints={'no_table': True})
    assert 'the table may not be used' in m.prompts[0]
    assert r['status'] == 'ok' and r['quality']['model_moves'] == 0
    assert not any(s['action'] == 'place-on-table' for s in r['plan'])


def test_old_endpoints_are_unchanged_and_compact_works(tmp_path):
    c = TestClient(create_app(openworld.RuleMock(), log_dir=tmp_path))
    old = c.post('/plan', json={'scene': SCENE, 'goal': GOAL}).json()
    assert old['status'] == 'ok' and 'model_plan' not in old and old['planner'] != 'model'
    small = c.post('/plan/model', json={'instruction': live_cases.SORT, 'scene': live_cases.four_towers(),
                                        'compact': True}).json()
    assert small['status'] == 'ok' and small['quality']['model_moves'] == 16 and small['quality']['planner_moves'] == 0


def test_ordered_goals_stay_with_the_planner(tmp_path):
    r = ask(Scripted([['b1', 'blue_zone']]), tmp_path, ordered=True)
    assert r['status'] == 'ok' and 'model_plan' not in r
    assert any('planned by the planner' in w for w in r['warnings'])


def test_bad_gate_is_refused(tmp_path):
    c = TestClient(create_app(Scripted([]), log_dir=tmp_path))
    assert c.post('/plan/model', json={'scene': SCENE, 'goal': GOAL, 'gate': 'maybe'}).status_code == 422
