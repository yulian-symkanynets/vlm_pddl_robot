"""Summarize benchmark runs into a table (printed + summary.md + summary.csv).

  python summarize.py results/20261106_101500                 # one run folder
  python summarize.py results/*  --by model mode views        # several folders, custom grouping
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

COLUMNS = [
    ('n', 'runs'),
    ('success_%', 'plan works in the real scene'),
    ('parsed_%', 'PDDL parsed (pddl/json/split modes)'),
    ('support_f1', 'perception: on/on-table/in facts vs gt'),
    ('support_exact_%', 'all support facts correct'),
    ('goal_exact_%', 'goal facts identical to gt'),
    ('lint_%', 'runs with inconsistent state'),
    ('optimal_%', 'success with shortest plan'),
    ('gen_s', 'median generation time'),
    ('plan_s', 'median planner time'),
    ('total_s', 'median end-to-end time'),
    ('out_tok', 'median output tokens'),
    ('vram_gb', 'max peak VRAM'),
]


def pct(xs):
    xs = list(xs)
    return round(100 * sum(bool(x) for x in xs) / len(xs), 1) if xs else None


def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(median(xs), 2) if xs else None


def summarize(rows):
    pdj = [r for r in rows if r['mode'] in ('pddl', 'json', 'split')]
    return {
        'n': len(rows),
        'success_%': pct(r.get('success') for r in rows),
        'parsed_%': pct(r.get('syntax_ok') for r in pdj) if pdj else None,
        'support_f1': round(mean(r.get('support_f1') or 0 for r in pdj), 3) if pdj else None,
        'support_exact_%': pct(r.get('support_exact') for r in pdj) if pdj else None,
        'goal_exact_%': pct(r.get('goal_exact') for r in pdj) if pdj else None,
        'lint_%': pct(r.get('lint_issues') for r in pdj) if pdj else None,
        'optimal_%': pct(r.get('optimal') for r in rows),
        'gen_s': med(r.get('generate_s') for r in rows),
        'plan_s': med(r.get('plan_s') for r in rows),
        'total_s': med(r.get('total_s') for r in rows),
        'out_tok': med(r.get('output_tokens') for r in rows),
        'vram_gb': max((r.get('peak_vram_gb') or 0 for r in rows), default=None),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folders', nargs='+')
    ap.add_argument('--by', nargs='+', default=['model', 'quant', 'mode'],
                    help='grouping keys, e.g. model mode views max_visual_tokens scene task')
    args = ap.parse_args()

    rows = []
    for f in args.folders:
        p = Path(f) / 'runs.jsonl'
        if p.exists():
            rows += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit('no runs.jsonl found')

    groups = defaultdict(list)
    for r in rows:
        groups[tuple(str(r.get(k)) for k in args.by)].append(r)
    header = list(args.by) + [c for c, _ in COLUMNS]
    table = [list(k) + [summarize(v)[c] for c, _ in COLUMNS] for k, v in sorted(groups.items())]

    fails = defaultdict(int)
    for r in rows:
        if not r.get('success'):
            fails[(r['model'], r['mode'], r.get('stage'), str(r.get('error'))[:70])] += 1

    md = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    md += ['| ' + ' | '.join('' if x is None else str(x) for x in row) + ' |' for row in table]
    md += ['', 'Columns: ' + '; '.join(f'**{c}** = {d}' for c, d in COLUMNS), '',
           '### Most common failures', '', '| model | mode | stage | error | count |', '|---|---|---|---|---|']
    md += [f'| {m} | {mo} | {s} | {e} | {n} |'
           for (m, mo, s, e), n in sorted(fails.items(), key=lambda x: -x[1])[:25]]
    text = '\n'.join(md)
    print(text)
    out = Path(args.folders[0])
    (out / 'summary.md').write_text(text + '\n')
    with (out / 'summary.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(table)
    print(f'\nwritten: {out / "summary.md"}, {out / "summary.csv"}')


if __name__ == '__main__':
    main()
