"""Score one VLM answer against the ground truth.

Pipeline (each stage can fail, `stage` says how far the answer got):
  extract  -> found a PDDL problem / JSON / plan in the text
  parse    -> PDDL parses against the domain (syntax, types, known predicates/objects)
  solve    -> a planner finds a plan for the VLM's problem
  execute  -> that plan, run on the GROUND-TRUTH problem, is applicable step by step and reaches the true goal
  success  -> all of the above

The key metric is `success`: would the plan have worked in the real scene?
A problem can parse and solve and still fail on the ground truth (wrong perception or wrong goal).
"""
from __future__ import annotations

import json
import time

from planning import (extract_json, extract_plan, extract_problem, goal_facts, init_facts,
                      lint_state, parse_problem, prf, simulate, solve, SUPPORT_PREDICATES)
from scenes import Scene, Task, gt_problem, make_problem

_GT_CACHE: dict = {}


def ground_truth(scene: Scene, task: Task, planner: str = 'auto') -> dict:
    key = (scene.id, task.task_id)
    if key not in _GT_CACHE:
        text = gt_problem(scene, task)
        problem = parse_problem(text)
        try:
            _, acts, _, _ = solve(problem, 'fast-downward-opt')
        except RuntimeError:
            _, acts, _, _ = solve(problem, planner)
        _GT_CACHE[key] = {
            'text': text, 'problem': problem, 'plan': acts,
            'plan_len': len(acts) if acts is not None else None,
            'support': {f for f in init_facts(problem) if f[0] in SUPPORT_PREDICATES},
            'goal': goal_facts(problem),
        }
    return _GT_CACHE[key]


def _problem_from_json(js: dict, scene: Scene, task: Task) -> tuple[str | None, str | None]:
    """Return (problem_text, error). Uses the same generator as the ground truth."""
    if not isinstance(js, dict) or not isinstance(js.get('support'), dict):
        return None, 'json without "support" mapping'
    support = {str(k): str(v) for k, v in js['support'].items()}
    if 'unknown' in support.values():
        return None, 'need_observation: unknown support for ' + \
            ', '.join(k for k, v in support.items() if v == 'unknown')
    goal = js.get('goal')
    if not isinstance(goal, list) or not all(isinstance(g, list) and g for g in goal):
        return None, 'json without a valid "goal" list'
    items = {i: scene.items.get(i, {}) for i in support}
    hidden = [h for h in js.get('hidden', []) if h in items]
    try:
        return make_problem(f'{scene.id}-{task.task_id}-vlm', scene.env, items, support,
                            [tuple(str(w) for w in g) for g in goal], hidden), None
    except ValueError as e:
        return None, f'invalid scene: {e}'


def evaluate(mode: str, text: str, scene: Scene, task: Task, planner: str = 'auto') -> dict:
    gt = ground_truth(scene, task, planner)
    r = {'stage': 'extract', 'success': False, 'error': None, 'gt_plan_len': gt['plan_len'],
         'problem_text': None, 'plan': None}

    # ---- plan mode (M1): execute the VLM's own plan on the ground truth
    if mode == 'plan':
        steps = extract_plan(text)
        if not steps:
            r['error'] = 'no actions found'
            return r
        r['plan'] = steps
        r['plan_len'] = len(steps)
        sim = simulate(gt['problem'], steps)
        r.update(stage='success' if sim['ok'] else 'execute', success=sim['ok'],
                 exec_fail_step=sim['fail_step'], error=sim['fail_reason'],
                 optimal=sim['ok'] and len(steps) == gt['plan_len'])
        return r

    # ---- pddl / json: get a problem text
    if mode == 'pddl':
        ptext = extract_problem(text)
        if ptext is None:
            r['error'] = 'no (define (problem ...)) found'
            return r
    elif mode == 'json':
        raw = extract_json(text)
        if raw is None:
            r['error'] = 'no JSON found'
            return r
        try:
            js = json.loads(raw)
        except json.JSONDecodeError as e:
            r['error'] = f'bad JSON: {e}'
            return r
        ptext, err = _problem_from_json(js, scene, task)
        if err:
            r['error'] = err
            r['need_observation'] = err.startswith('need_observation')
            return r
    else:
        raise ValueError(mode)
    r['problem_text'] = ptext

    # ---- parse
    r['stage'] = 'parse'
    try:
        t0 = time.perf_counter()
        problem = parse_problem(ptext)
        r['parse_s'] = round(time.perf_counter() - t0, 3)
    except Exception as e:
        r['error'] = f'parse: {type(e).__name__}: {str(e)[:300]}'
        return r
    r['syntax_ok'] = True

    # ---- compare with ground truth (perception + goal understanding)
    issues = lint_state(problem)
    r['lint_issues'] = issues
    support = {f for f in init_facts(problem) if f[0] in SUPPORT_PREDICATES}
    s = prf(support, gt['support'])
    r['support_f1'] = s['f1']
    r['support_exact'] = support == gt['support']
    r['support_missing'] = sorted(' '.join(f) for f in gt['support'] - support)
    r['support_extra'] = sorted(' '.join(f) for f in support - gt['support'])
    goal = goal_facts(problem)
    r['goal_exact'] = goal == gt['goal']
    r['goal_f1'] = prf(goal, gt['goal'])['f1']
    r['goal_missing'] = sorted(' '.join(f) for f in gt['goal'] - goal)
    r['goal_extra'] = sorted(' '.join(f) for f in goal - gt['goal'])

    # ---- plan for the VLM's problem
    r['stage'] = 'solve'
    try:
        status, acts, dt, used = solve(problem, planner)
    except Exception as e:
        r['error'] = f'planner: {type(e).__name__}: {str(e)[:300]}'
        return r
    r.update(planner=used, plan_s=round(dt, 3), planner_status=status)
    if acts is None:
        r['error'] = f'no plan ({status})'
        return r
    r['plan'] = acts
    r['plan_len'] = len(acts)

    # ---- does that plan work in the real (ground-truth) scene?
    r['stage'] = 'execute'
    sim = simulate(gt['problem'], acts)
    r['exec_fail_step'] = sim['fail_step']
    if not sim['ok']:
        r['error'] = f'on ground truth: {sim["fail_reason"]}'
        return r
    r.update(stage='success', success=True, optimal=len(acts) == gt['plan_len'])
    return r
