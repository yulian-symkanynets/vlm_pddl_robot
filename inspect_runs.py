"""Explain each run of a benchmark folder: why did it fail?

  python inspect_runs.py results/20261106_104603              # all runs
  python inspect_runs.py results/20261106_104603 --failed     # only failures
  python inspect_runs.py results/20261106_104603 --answer     # also print the model's raw answer

For every run it prints: the error, which support facts the model got wrong (missing / extra vs.
ground truth), goal differences, state inconsistencies (lint), the plan, and a one-line diagnosis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def diagnose(r: dict) -> str:
    stage, err = r.get('stage'), str(r.get('error') or '')
    if r.get('success'):
        return 'OK' + ('' if r.get('optimal') else ' (plan longer than optimal)')
    if stage == 'generate':
        return 'generation crashed (see error)'
    if stage == 'extract':
        return 'answer did not contain a usable PDDL problem / JSON / action list (format problem)'
    if stage == 'parse':
        return 'PDDL syntax or naming error (unknown predicate / object / type)'
    if r.get('mode') == 'plan':
        return ('the model planned by itself and broke a precondition' if 'precondition' in err
                else 'the model planned by itself and did not reach the goal')
    parts = []
    if r.get('support_missing') or r.get('support_extra'):
        parts.append('PERCEPTION: wrong support relations')
    if r.get('lint_issues'):
        parts.append('BOOKKEEPING: inconsistent state (e.g. clear/on contradict each other)')
    if r.get('goal_missing') or r.get('goal_extra'):
        parts.append('TASK: goal misunderstood')
    if stage == 'solve':
        parts.append('planner found no plan for the model\'s problem')
    return '; '.join(parts) or 'unclear: compare the raw answer with the ground truth'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder')
    ap.add_argument('--failed', action='store_true')
    ap.add_argument('--answer', action='store_true', help='print the raw model answer too')
    args = ap.parse_args()
    folder = Path(args.folder)
    rows = [json.loads(l) for l in (folder / 'runs.jsonl').read_text().splitlines() if l.strip()]
    for r in rows:
        if args.failed and r.get('success'):
            continue
        print('=' * 100)
        print(f"{r['run_id']}")
        print(f"  mode={r['mode']} views={r['views']} stage={r.get('stage')} success={r.get('success')} "
              f"gen={r.get('generate_s')}s out_tokens={r.get('output_tokens')} truncated={r.get('truncated')}")
        print(f"  DIAGNOSIS: {diagnose(r)}")
        if r.get('error'):
            print(f"  error: {r['error']}")
        for key, title in (('support_missing', 'gt support facts the model MISSED'),
                           ('support_extra', 'support facts the model INVENTED'),
                           ('goal_missing', 'gt goal facts MISSING'),
                           ('goal_extra', 'goal facts INVENTED'),
                           ('lint_issues', 'state inconsistencies')):
            if r.get(key):
                print(f"  {title}:")
                for x in r[key]:
                    print(f"      {x}")
        if r.get('plan'):
            fail = r.get('exec_fail_step')
            print(f"  plan ({r.get('plan_len')} steps, optimal {r.get('gt_plan_len')}):")
            for k, (a, p) in enumerate(r['plan']):
                mark = '  <-- FAILS HERE on the real scene' if fail == k else ''
                print(f"      {k + 1}. ({a} {' '.join(p)}){mark}")
        if args.answer:
            raw = folder / 'raw' / f"{r['run_id']}.txt"
            if raw.exists():
                txt = raw.read_text()
                ans = txt.split('### ANSWER', 1)[1].split('### PROBLEM USED', 1)[0].strip()
                print('  ---- model answer ----')
                print('\n'.join('  | ' + l for l in ans.splitlines()[:80]))
    print('=' * 100)
    ok = sum(1 for r in rows if r.get('success'))
    print(f'{ok}/{len(rows)} successful')


if __name__ == '__main__':
    main()
