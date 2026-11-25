"""Measure every number that docs/how_it_works.pdf quotes, and write results/how_it_works.json.

    python scripts/measure_how_it_works.py        # about a minute on a laptop, no GPU, no model

Goals are given directly (or read by the rule-based stand-in model), so these are planner and
server numbers only. Model times come from the live benchmark on the GPU machine (live_bench.py).
"""
from __future__ import annotations

import copy
import datetime as dt
import gzip
import json
import platform
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import formula_plan       # noqa: E402
import goal_logic         # noqa: E402
import live_cases         # noqa: E402
import openworld          # noqa: E402

FOUR = live_cases.four_towers()


def timed(**kw):
    t = time.perf_counter()
    r = openworld.run(**kw)
    return r, round(time.perf_counter() - t, 3)


def summary(r, wall):
    q = r.get('quality') or {}
    return {'status': r['status'], 'planner': r.get('planner'), 'moves': q.get('moves'),
            'lower_bound': q.get('lower_bound'), 'optimal': q.get('optimal'), 'proof': q.get('proof'),
            'plan_s': (r.get('timings') or {}).get('plan_s'), 'wall_s': wall, 'message': r.get('message')}


def goal_types():
    no_table = {'no_table': True}
    cases = [
        ('sort_by_color', 'every object into the zone of its colour', dict(goal=[{'sort_by_color': True}])),
        ('sort_by_color + no_table', 'the same, the table may not be used',
         dict(goal=[{'sort_by_color': True}], constraints=no_table)),
        ('stack_by_color', 'one stack per colour, anywhere on the table', dict(goal=[{'stack_by_color': True}])),
        ('stack_in_zones + no_table', 'one tower per colour in its zone, no table',
         dict(goal=[{'stack_in_zones': True}], constraints=no_table)),
        ('formula', 'every cube into the red zone',
         dict(goal=[{'formula': live_cases.REF_CUBES_RED}])),
        ('ordered (3 goals)', 'first green cube 2 on the table, then red block 1 on it, then blue block 2 on that',
         dict(goal=[{'object': 'green_cube_2', 'on': 'table'}, {'object': 'red_block_1', 'on': 'green_cube_2'},
                    {'object': 'blue_block_2', 'on': 'red_block_1'}], ordered=True)),
    ]
    out = []
    for name, words, kw in cases:
        r, wall = timed(scene=FOUR, **kw)
        out.append({'name': name, 'words': words, **summary(r, wall)})
    return out


def example_request():
    req = {'instruction': live_cases.SORT, 'scene': FOUR}
    r, wall = timed(scene=FOUR, instruction=live_cases.SORT, vlm=openworld.RuleMock())
    photos = sorted((ROOT / 'images').glob('*.jpg'))
    return {'request_bytes': len(json.dumps(req)), 'scene_bytes': len(json.dumps(FOUR)),
            'photo_bytes_median': sorted(p.stat().st_size for p in photos)[len(photos) // 2] if photos else None,
            'photos': len(photos), 'goal_answer': r['model_answers'].get('goal'), 'goal': r.get('goal'),
            'first_steps': [s['pddl'] for s in r['plan'][:6]], 'steps': len(r['plan']), **summary(r, wall)}


def scaling(sizes=(4, 8, 16, 24, 32, 48)):
    rows = []
    for n in sizes:
        sc = live_cases.towers_scene(n)
        for name, goal in (('sort_by_color', [{'sort_by_color': True}]), ('stack_by_color', [{'stack_by_color': True}]),
                           ('stack as formula', [{'formula': goal_logic.STACK_BY_COLOUR}])):
            r, wall = timed(scene=sc, goal=goal)
            rows.append({'n': n, 'goal': name, **summary(r, wall)})
    return rows


def farby_like(seed):
    rnd = random.Random(seed)
    cols = ['red', 'yellow', 'blue', 'green']
    objs = [{'id': f'{c}_plank', 'color': c, 'size_mm': 80, 'height_mm': 20, 'on': f'{c}_zone'} for c in cols]
    rest = []
    for c in cols:
        rest += [{'id': f'{c}_cube', 'color': c, 'size_mm': 40, 'height_mm': 40},
                 {'id': f'{c}_cyl', 'color': c, 'size_mm': 30, 'height_mm': 40},
                 {'id': f'{c}_bar', 'color': c, 'size_mm': 25, 'height_mm': 20}]
    rnd.shuffle(rest)
    tops = {}
    for k, o in enumerate(rest):
        o['on'] = tops.get(k % 4, 'table')
        tops[k % 4] = o['id']
    return {'objects': objs + rest,
            'zones': [{'id': f'{c}_zone', 'color': c, 'capacity': 1, 'max_height_mm': 140} for c in cols]}


def greedy_vs_exact(seeds=range(8)):
    cons = {'no_table': True, 'keep': [f'{c}_plank' for c in ('red', 'yellow', 'blue', 'green')]}
    rows = []
    for seed in seeds:
        scene = farby_like(seed)
        saved = formula_plan.EXACT_NODES
        formula_plan.EXACT_NODES = 0
        g, gw = timed(scene=scene, goal=[{'stack_in_zones': True}], constraints=cons)
        formula_plan.EXACT_NODES = saved
        e, ew = timed(scene=scene, goal=[{'stack_in_zones': True}], constraints=cons)
        ok = openworld.check_plan(scene, [{'stack_in_zones': True}], e.get('plan') or [], cons)['status']
        rows.append({'seed': seed, 'lower_bound': (e.get('quality') or {}).get('lower_bound'),
                     'greedy_moves': (g.get('quality') or {}).get('moves'), 'greedy_s': gw,
                     'exact_moves': (e.get('quality') or {}).get('moves'), 'exact_s': ew,
                     'exact_optimal': (e.get('quality') or {}).get('optimal'), 'check': ok})
    return rows


def wire_sizes():
    from fastapi.testclient import TestClient
    from server import create_app
    c = TestClient(create_app(openworld.RuleMock(), log_dir=Path(tempfile.mkdtemp())))
    body = {'instruction': live_cases.SORT, 'scene': FOUR}
    full = c.post('/plan', json=body)
    small = c.post('/plan', json={**body, 'compact': True})
    raw = json.dumps({**body, 'compact': True}).encode()
    zipped = c.post('/plan', content=gzip.compress(raw), headers={'Content-Type': 'application/json',
                                                                  'Content-Encoding': 'gzip', 'Accept-Encoding': 'gzip'})
    return {'request_bytes': len(raw), 'request_gzip_bytes': len(gzip.compress(raw)),
            'answer_full_bytes': len(full.content), 'answer_compact_bytes': len(small.content),
            'answer_compact_gzip_bytes': int(zipped.headers['content-length']),
            'compact_example': small.json(), 'x_server_ms': float(zipped.headers['x-server-ms'])}


def replanning():
    first, _ = timed(scene=FOUR, goal=[{'sort_by_color': True}])
    after = copy.deepcopy(FOUR)
    step = first['plan'][1]
    for o in after['objects']:
        if o['id'] == step['args'][0]:
            o['on'] = step['args'][-1]
    t = time.perf_counter()
    kept = openworld.replan(first, after)
    kept_s = round(time.perf_counter() - t, 3)
    moved = copy.deepcopy(FOUR)
    for o in moved['objects']:                               # a person puts a cube on the table
        if o['id'] == 'green_cube_2':
            o['on'] = 'table'
    t = time.perf_counter()
    re = openworld.replan(first, moved)
    re_s = round(time.perf_counter() - t, 3)
    small = {'objects': [{'id': 'r1', 'color': 'red', 'on': 'table'}, {'id': 'b1', 'color': 'blue', 'on': 'r1'},
                         {'id': 'g1', 'color': 'green', 'on': 'table'}],
             'zones': [{'id': 'red_zone', 'color': 'red', 'capacity': 1}]}
    base, _ = timed(scene=small, goal=[{'object': 'r1', 'in': 'red_zone'}])
    failed = openworld.replan(base, small, failed=[base['plan'][1]['pddl']])
    return {'first_moves': first['quality']['moves'],
            'kept': {'decision': kept['decision'], 'moves_left': kept['quality']['moves'], 'seconds': kept_s,
                     'proof': kept['quality'].get('proof')},
            'changed': {'decision': re['decision'], 'changes': re['changes'], 'moves': re['quality']['moves'],
                        'seconds': re_s},
            'failed': {'first_plan': [s['pddl'] for s in base['plan']], 'reported': base['plan'][1]['pddl'],
                       'new_plan': [s['pddl'] for s in failed['plan']], 'rules': failed.get('failed_rules'),
                       'constraints': failed.get('constraints')}}


def checking():
    scene = farby_like(4)
    cons = {'no_table': True, 'keep': [f'{c}_plank' for c in ('red', 'yellow', 'blue', 'green')]}
    sup = {o['id']: o['on'] for o in scene['objects']}
    top = next(x for x in sup if x not in sup.values() and sup[x] != 'table' and not sup[x].endswith('_zone'))
    bad = [f'pick-from-item {top} {sup[top]}', f'place-on-table {top}']
    big = next(o for o in scene['objects'] if o['id'].endswith('_cube'))['id']
    return {'table': openworld.check_plan(scene, [{'stack_in_zones': True}], bad, cons),
            'table_plan': bad, 'note': f'{big} is 40 mm, bars are 25 mm'}


def main():
    t0 = time.time()
    out = {'measured': dt.datetime.now().isoformat(timespec='seconds'),
           'machine': f'{platform.system()} {platform.machine()}, Python {platform.python_version()}',
           'model': 'none (goals given or read by the rule-based stand-in)',
           'example': example_request(), 'goal_types': goal_types(), 'scaling': scaling(),
           'greedy_vs_exact': greedy_vs_exact(), 'wire': wire_sizes(), 'replanning': replanning(),
           'checking': checking()}
    out['seconds'] = round(time.time() - t0, 1)
    path = ROOT / 'results' / 'how_it_works.json'
    path.write_text(json.dumps(out, indent=1, default=str))
    print(f'wrote {path} in {out["seconds"]} s')


if __name__ == '__main__':
    main()
