"""Error decomposition: re-score saved answers with one part replaced by the ground truth.

  python rescore.py results/20261106_104603

For every pddl/json answer three variants are evaluated (no model needed, uses raw/*.txt):
  as_is           the model's answer unchanged
  oracle_goal     model's scene description + CORRECT goal       -> is perception good enough?
  oracle_scene    CORRECT scene + model's goal                    -> is goal understanding good enough?
If oracle_goal succeeds, the failure was only the goal. If oracle_scene succeeds, it was only perception.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from evaluate import evaluate, ground_truth, merge_split
from planning import _balanced, extract_json, extract_problem
from scenes import load_config


def goal_section(problem_text: str) -> str | None:
    m = re.search(r'\(\s*:goal', problem_text)
    return _balanced(problem_text, m.start(), '(', ')') if m else None


def swap_goal(problem_text: str, new_goal_section: str) -> str:
    old = goal_section(problem_text)
    return problem_text.replace(old, new_goal_section) if old else problem_text


def variants(mode: str, answer: str, scene, task) -> dict[str, str | None]:
    gt = ground_truth(scene, task)
    gt_goal_sec = goal_section(gt['text'])
    if mode == 'pddl':
        prob = extract_problem(answer)
        if not prob:
            return {'as_is': answer, 'oracle_goal': None, 'oracle_scene': None}
        model_goal = goal_section(prob)
        return {'as_is': answer,
                'oracle_goal': swap_goal(prob, gt_goal_sec),
                'oracle_scene': swap_goal(gt['text'], model_goal) if model_goal else None}
    if mode in ('json', 'split'):
        if mode == 'split':
            merged, _ = merge_split(answer)
            if merged is None:
                return {'as_is': answer, 'oracle_goal': None, 'oracle_scene': None}
            answer = merged
            mode = 'json'
        raw = extract_json(answer)
        try:
            js = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            js = None
        if not isinstance(js, dict):
            return {'as_is': answer, 'oracle_goal': None, 'oracle_scene': None}
        a = dict(js, goal=[list(g) for g in task.goal])
        b = dict(js, support=dict(scene.support), hidden=list(scene.hidden))
        return {'as_is': json.dumps(js), 'oracle_goal': json.dumps(a), 'oracle_scene': json.dumps(b)}
    return {'as_is': answer}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder')
    ap.add_argument('--verbose', action='store_true', help='print the model goal / support per run')
    args = ap.parse_args()
    folder = Path(args.folder)
    cfg = load_config()
    rows = [json.loads(l) for l in (folder / 'runs.jsonl').read_text().splitlines() if l.strip()]
    table = defaultdict(lambda: defaultdict(list))
    out_lines = []
    for r in rows:
        raw = folder / 'raw' / f"{r['run_id']}.txt"
        if not raw.exists():
            continue
        answer = raw.read_text().split('### ANSWER', 1)[1].split('### PROBLEM USED', 1)[0].strip()
        scene = cfg.scene(r['scene'])
        task = next(t for t in scene.tasks if t.task_id == r['task'])
        res = {}
        emode = 'json' if r['mode'] == 'split' else r['mode']
        for name, text in variants(r['mode'], answer, scene, task).items():
            ok = bool(text) and evaluate(emode, text, scene, task)['success']
            res[name] = ok
            table[(r['model'], r['mode'])][name].append(ok)
        out_lines.append(f"{r['run_id']:70s} " + '  '.join(f'{k}={"OK" if v else "--"}' for k, v in res.items()))
        if args.verbose and r['mode'] in ('pddl', 'json', 'split'):
            ev = evaluate(r['mode'], answer, scene, task)
            out_lines.append(f"      support missed={ev.get('support_missing')} invented={ev.get('support_extra')}")
            out_lines.append(f"      goal    missed={ev.get('goal_missing')} invented={ev.get('goal_extra')}")
    print('\n'.join(out_lines))
    print('\n| model | mode | n | as_is % | oracle_goal % (perception ok?) | oracle_scene % (goal ok?) |')
    print('|---|---|---|---|---|---|')
    for (model, mode), d in sorted(table.items()):
        def pct(k):
            v = d.get(k)
            return f'{100 * sum(v) / len(v):.0f}' if v else '-'
        print(f'| {model} | {mode} | {len(d["as_is"])} | {pct("as_is")} | {pct("oracle_goal")} | {pct("oracle_scene")} |')


if __name__ == '__main__':
    main()
