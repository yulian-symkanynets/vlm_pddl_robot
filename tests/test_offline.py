"""Offline tests (no GPU, no model). Run: python -m pytest tests -q"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from evaluate import evaluate, ground_truth
from planning import extract_plan, extract_problem, parse_problem, simulate, strip_reasoning
from scenes import load_config

CFG = load_config()
S5 = CFG.scene('s5_reorder')
SWAP = S5.tasks[0]


@pytest.mark.parametrize('scene,task', [(s, t) for s in CFG.scenes for t in s.tasks],
                         ids=lambda x: getattr(x, 'id', getattr(x, 'task_id', '')))
def test_ground_truth_is_solvable(scene, task):
    gt = ground_truth(scene, task)
    assert gt['plan_len'] and simulate(gt['problem'], gt['plan'])['ok']


def test_swap_stacks_optimal_length():
    assert ground_truth(S5, SWAP)['plan_len'] == 6        # 3 pick + 3 place


def test_perfect_pddl_answer_succeeds():
    text = 'Sure!\n```pddl\n' + ground_truth(S5, SWAP)['text'] + '```'
    r = evaluate('pddl', text, S5, SWAP)
    assert r['success'] and r['support_exact'] and r['goal_exact'] and r['optimal']


def test_reasoning_tags_are_stripped():
    text = '<think>R is on Y (on R Y) ...</think>\n```pddl\n' + ground_truth(S5, SWAP)['text'] + '```'
    assert extract_problem(text).startswith('(define')
    assert strip_reasoning('<think>x</think><answer>ok</answer>') == 'ok'


def test_missed_stack_fails_on_ground_truth():
    """VLM says the red cylinder stands on the table -> its plan picks the covered yellow block."""
    text = ground_truth(S5, SWAP)['text'].replace('(on red_cylinder yellow_block)',
                                                  '(on-table red_cylinder)')
    text = text.replace('(on-table yellow_block)', '(on-table yellow_block)\n    (clear yellow_block)')
    r = evaluate('pddl', text, S5, SWAP)
    assert r['stage'] == 'execute' and not r['success']
    assert 'precondition violated' in r['error']
    assert not r['support_exact']


def test_inconsistent_clear_is_linted():
    text = ground_truth(S5, SWAP)['text'].replace('(hand-empty)', '(hand-empty)\n    (clear yellow_block)')
    r = evaluate('pddl', text, S5, SWAP)
    assert any('clear(yellow_block)' in i for i in r['lint_issues'])


def test_unknown_predicate_fails_parse():
    text = ground_truth(S5, SWAP)['text'].replace('(hand-empty)', '(hand-empty)\n    (stacked red_cylinder)')
    r = evaluate('pddl', text, S5, SWAP)
    assert r['stage'] == 'parse' and not r['success']


def test_unbalanced_parentheses_fail_extract():
    text = ground_truth(S5, SWAP)['text'].rstrip()[:-1]
    assert evaluate('pddl', text, S5, SWAP)['stage'] == 'extract'


def test_plan_that_picks_bottom_block_fails():
    plan = '(pick-from-table yellow_block)\n(place-in yellow_block yellow_zone_s1 yellow_zone)'
    r = evaluate('plan', plan, S5, SWAP)
    assert not r['success'] and r['exec_fail_step'] == 0


def test_correct_plan_in_function_syntax():
    plan = """1. pick-from-item(red_cylinder, yellow_block)
2. place-on-table(red_cylinder)
3. pick-from-item(green_block, blue_cylinder)
4. place-on-item(green_block, red_cylinder)
5. pick-from-table(blue_cylinder)
6. place-on-item(blue_cylinder, yellow_block)"""
    assert len(extract_plan(plan)) == 6
    assert evaluate('plan', plan, S5, SWAP)['success']


def test_json_mode():
    js = {'support': dict(S5.support), 'hidden': [], 'goal': [list(g) for g in SWAP.goal]}
    assert evaluate('json', json.dumps(js), S5, SWAP)['success']
    js['support']['red_cylinder'] = 'unknown'
    r = evaluate('json', json.dumps(js), S5, SWAP)
    assert r.get('need_observation') and not r['success']


def test_json_two_items_on_one_is_rejected():
    js = {'support': {**S5.support, 'green_block': 'yellow_block'}, 'goal': [list(g) for g in SWAP.goal]}
    r = evaluate('json', json.dumps(js), S5, SWAP)
    assert 'invalid scene' in r['error']


def test_closed_box_needs_open_lid():
    env = {'places': {**S5.env['places'], 'box_1': {'kind': 'box', 'slots': 2, 'open': False}}}
    from scenes import make_problem
    from planning import solve
    p = parse_problem(make_problem('box', env, S5.items, S5.support, [('in', 'yellow_block', 'box_1')]))
    _, acts, _, _ = solve(p)
    assert acts[0][0] == 'open-lid'
    assert ('pick-from-item', ['red_cylinder', 'yellow_block']) in acts


def test_too_tall_for_shelf_is_unsolvable():
    env = {'places': {'shelf_low': {'kind': 'shelf', 'slots': 2, 'open': True, 'clearance_mm': 50}}}
    from scenes import make_problem
    from planning import solve
    p = parse_problem(make_problem('shelf', env, S5.items, S5.support, [('in', 'yellow_block', 'shelf_low')]))
    status, acts, _, _ = solve(p)
    assert acts is None
