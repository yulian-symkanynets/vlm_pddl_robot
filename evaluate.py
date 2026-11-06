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
from prompts import SPLIT_SEP
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


GOAL_PREDICATES = {'on': 3, 'in': 3, 'on-table': 2}


def normalize_goal(goal) -> list[tuple[str, ...]] | None:
    """Accept ["on","x","y"], ["x","on","y"] lists or named fields {"object": x, "on"/"in": y}.

    Returns None for anything else, so a malformed goal is reported as a goal error
    instead of producing broken PDDL that only fails later in the parser.
    """
    if not isinstance(goal, list) or not goal:
        return None
    out = []
    for g in goal:
        if isinstance(g, list) and g:
            t = [str(w) for w in g]
            if t[0] not in GOAL_PREDICATES and len(t) == 3 and t[1] in GOAL_PREDICATES:
                t = [t[1], t[0], t[2]]                 # ["x", "on", "y"] -> ["on", "x", "y"]
            if t[0] == 'on' and len(t) == 3 and t[2] == 'table':
                t = ['on-table', t[1]]
            if t[0] not in GOAL_PREDICATES or len(t) != GOAL_PREDICATES[t[0]]:
                return None
            out.append(tuple(t))
        elif isinstance(g, dict) and 'object' in g:
            x = str(g['object'])
            if 'in' in g:
                out.append(('in', x, str(g['in'])))
            elif str(g.get('on')) == 'table':
                out.append(('on-table', x))
            elif 'on' in g:
                out.append(('on', x, str(g['on'])))
            else:
                return None
        else:
            return None
    return out


def merge_split(text: str) -> tuple[str | None, str | None]:
    """Split mode: combine the scene answer and the goal answer into one json answer."""
    scene_txt, _, goal_txt = text.partition(SPLIT_SEP)
    out = {}
    for name, part in (('scene', scene_txt), ('goal', goal_txt)):
        raw = extract_json(part)
        if raw is None:
            return None, f'no JSON in the {name} answer'
        try:
            out.update(json.loads(raw))
        except json.JSONDecodeError as e:
            return None, f'bad JSON in the {name} answer: {e}'
    return json.dumps(out), None


def _problem_from_json(js: dict, scene: Scene, task: Task) -> tuple[str | None, str | None]:
    """Return (problem_text, error). Uses the same generator as the ground truth."""
    if not isinstance(js, dict) or not isinstance(js.get('support'), dict):
        return None, 'json without "support" mapping'
    support = {str(k): str(v) for k, v in js['support'].items()}
    if 'unknown' in support.values():
        return None, 'need_observation: unknown support for ' + \
            ', '.join(k for k, v in support.items() if v == 'unknown')
    goal = normalize_goal(js.get('goal'))
    if goal is None:
        return None, f'invalid goal: {json.dumps(js.get("goal"))[:200]}'
    items = {i: scene.items.get(i, {}) for i in support}
    known = set(items) | set(scene.env['places'])
    unknown = sorted({w for g in goal for w in g[1:] if w not in known})
    if unknown:
        return None, f'goal uses unknown names: {", ".join(unknown)}'
    hidden = [h for h in js.get('hidden', []) if h in items]
    try:
        return make_problem(f'{scene.id}-{task.task_id}-vlm', scene.env, items, support,
                            goal, hidden), None
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
    elif mode in ('json', 'split'):
        if mode == 'split':
            text, err = merge_split(text)
            if err:
                r['error'] = err
                return r
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
