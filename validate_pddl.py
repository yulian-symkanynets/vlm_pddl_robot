"""Validate a PDDL problem (or a plan) by hand, without running a VLM.

  # show the ground-truth problem for a scene/task
  python validate_pddl.py --scene s5_reorder --task swap_stacks --print-gt

  # check a problem file (e.g. copied from a VLM answer): parse, consistency, solve
  python validate_pddl.py my_problem.pddl

  # ... and check whether its plan works in the real scene
  python validate_pddl.py my_problem.pddl --scene s5_reorder --task swap_stacks

  # check a plan written by a VLM (one action per line) against the real scene
  python validate_pddl.py --plan my_plan.txt --scene s5_reorder --task swap_stacks

A VLM answer file can be passed directly; the problem / plan is extracted from the text.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate import evaluate, ground_truth
from planning import extract_plan, extract_problem, lint_state, parse_problem, solve
from scenes import load_config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('problem', nargs='?', help='problem .pddl or a text file containing one')
    ap.add_argument('--plan', help='plan / answer text file to execute on the ground truth')
    ap.add_argument('--scene')
    ap.add_argument('--task')
    ap.add_argument('--print-gt', action='store_true')
    ap.add_argument('--planner', default='auto')
    args = ap.parse_args()

    scene = task = None
    if args.scene:
        cfg = load_config()
        scene = cfg.scene(args.scene)
        task = next(t for t in scene.tasks if t.task_id == (args.task or scene.tasks[0].task_id))

    if args.print_gt:
        gt = ground_truth(scene, task, args.planner)
        print(gt['text'])
        print('; optimal plan (%d steps):' % gt['plan_len'])
        for a, p in gt['plan']:
            print(f';   ({a} {" ".join(p)})')
        return

    if args.plan:
        if not scene:
            raise SystemExit('--plan needs --scene (and --task)')
        text = Path(args.plan).read_text()
        print('actions found:', len(extract_plan(text)))
        print(json.dumps(evaluate('plan', text, scene, task, args.planner), indent=2, default=str))
        return

    if not args.problem:
        ap.error('give a problem file, --plan, or --print-gt')
    text = Path(args.problem).read_text()
    if scene:
        r = evaluate('pddl', text, scene, task, args.planner)
        r.pop('problem_text', None)
        print(json.dumps(r, indent=2, default=str))
        return

    ptext = extract_problem(text)
    if not ptext:
        raise SystemExit('no (define (problem ...)) found')
    try:
        problem = parse_problem(ptext)
    except Exception as e:
        raise SystemExit(f'PARSE ERROR: {type(e).__name__}: {e}')
    print('parse: OK')
    issues = lint_state(problem)
    print('consistency:', 'OK' if not issues else '')
    for i in issues:
        print('  -', i)
    status, acts, dt, used = solve(problem, args.planner)
    print(f'planner {used}: {status} in {dt:.2f} s')
    for a, p in acts or []:
        print(f'  ({a} {" ".join(p)})')


if __name__ == '__main__':
    main()
