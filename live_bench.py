#!/usr/bin/env python
"""Live benchmark: run real requests against a running server and measure everything.

    ./serve.sh start qwen3-vl-4b 4bit            # on HEKATE (see SERVER.md); --no-memory for clean timings
    python live_bench.py                         # all suites against http://127.0.0.1:8010
    python live_bench.py --quick                 # about a fifth of the cases, a few minutes
    python live_bench.py --suites json,scale     # no photos
    python live_bench.py --url http://192.168.73.15:8010 --label via-lan
    python live_bench.py --photos live_photos    # your own photos (name.jpg + name.json true scene)
    python live_report.py results/live/<run> results/live/<other run>    # compare models / runs

Suites
    json    sentences + scene JSON -> /plan.   The model only reads the sentence (goal), so this
            measures language understanding and the whole server, per kind of task and scene size.
    scale   no model: /solve with a given goal at growing scene sizes (planner time and plan length)
    photo   photos with a known true scene: /perceive at every resolution (what is seen, how long),
            then /plan from the photo and from the true JSON for the same task (semantic route
            against image route: time, bytes sent, success)

A request counts as correct only when the plan really works: it is replayed here on the TRUE scene
with the benchmark's own rules and the final arrangement must satisfy a reference written by a
person (live_score.py). "wrong plan" means the server said ok but the plan does not do the task.

Output: results/live/<time>_<label>/{results.json, rows.csv, report.html, charts/*.png}
Every number in the report comes from results.json.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import statistics
import sys
import time
from pathlib import Path

import live_cases
import live_score

ROOT = Path(__file__).resolve().parent
TIMEOUT_S = 900
RESOLUTIONS = 'low,medium,high,full'


class Http:
    """httpx client with a base URL (httpx is in requirements.txt). Tests pass a FastAPI TestClient
    instead: it has the same get/post signature."""

    def __init__(self, url: str):
        import httpx
        self.c = httpx.Client(base_url=url.rstrip('/'), timeout=TIMEOUT_S)

    def get(self, path, **kw):
        return self.c.get(path, **kw)

    def post(self, path, **kw):
        return self.c.post(path, **kw)


def call(api, method: str, path: str, **kw) -> tuple[int, dict, float]:
    """-> (http status, JSON body, seconds as seen by the client)."""
    t0 = time.perf_counter()
    try:
        r = getattr(api, method)(path, **kw)
        code, text = r.status_code, r.text
    except Exception as e:                                   # noqa: BLE001  (connection, timeout ...)
        return 0, {'status': 'error', 'message': f'{type(e).__name__}: {e}'}, time.perf_counter() - t0
    wall = time.perf_counter() - t0
    try:
        body = json.loads(text)
    except ValueError:
        body = {'status': 'error', 'message': text[:200]}
    if code != 200 and 'status' not in body:
        body = {'status': 'error', 'message': str(body.get('detail', body))[:200]}
    return code, body, wall


# --------------------------------------------------------------------------- rows
def goal_text(body: dict) -> str:
    g = body.get('goal')
    if not g:
        return ''
    first = g[0] if isinstance(g, list) and g else {}
    if isinstance(first, dict) and first.get('in_words'):
        return first['in_words']
    if isinstance(first, dict) and any(k in first for k in ('sort_by_color', 'stack_by_color')):
        return 'sort by colour' if first.get('sort_by_color') else 'one stack per colour'
    return f'{len(g)} entries: ' + '; '.join(json.dumps(x) for x in g[:3]) + (' ...' if len(g) > 3 else '')


def make_row(case: dict, route: str, code: int, body: dict, wall: float, verdict: dict,
             sent_bytes: int, **extra) -> dict:
    t, q, mem = body.get('timings') or {}, body.get('quality') or {}, body.get('memory') or {}
    res = body.get('resolution') or {}
    return {
        'suite': case['suite'], 'group': case['group'], 'name': case['name'], 'route': route,
        'n_objects': case.get('n_objects'), 'instruction': case.get('instruction', ''),
        'expected': case.get('expect') or ['ok'], 'status': body.get('status'),
        'outcome': verdict['outcome'], 'passed': verdict['passed'], 'reasons': verdict['reasons'],
        'http': code, 'wall_s': round(wall, 3), 'perceive_s': t.get('perceive_s'), 'goal_s': t.get('goal_s'),
        'plan_s': t.get('plan_s'), 'total_s': t.get('total_s'), 'moves': q.get('moves'),
        'optimal': q.get('optimal'), 'planner': body.get('planner'), 'attempts': len(body.get('attempts') or []) or None,
        'memory_goal': mem.get('goal'), 'memory_plan': mem.get('plan'), 'sent_bytes': sent_bytes,
        'goal_text': goal_text(body), 'message': (body.get('message') or '')[:300],
        'resolution': res.get('used') if isinstance(res, dict) else None,
        'levels_tried': len(res.get('tried') or []) or None if isinstance(res, dict) else None,
        'approx_tokens': sum(t.get('approx_tokens', 0) for t in res.get('tried') or []) or None if isinstance(res, dict) else None,
        **extra,
    }


# --------------------------------------------------------------------------- suites
def run_json(api, cases, emit):
    for c in cases:
        body = {'instruction': c['instruction'], 'scene': c['scene']}
        if c.get('goal'):
            body['goal'] = c['goal']
        code, resp, wall = call(api, 'post', '/plan', json=body)
        truth = c.get('truth') or c['scene']
        v = live_score.judge(truth, resp, c.get('expect'), c.get('reference'), c.get('check') or None)
        emit(make_row(c, 'json', code, resp, wall, v, len(json.dumps(c['scene']))))


def run_scale(api, cases, emit):
    for c in cases:
        code, resp, wall = call(api, 'post', '/solve', json={'scene': c['scene'], 'goal': c['goal']})
        v = live_score.judge(c['scene'], resp, c.get('expect'), c.get('reference'))
        emit(make_row(c, 'json', code, resp, wall, v, len(json.dumps(c['scene']))))


def _mime(p: Path) -> str:
    return 'image/png' if p.suffix.lower() == '.png' else 'image/jpeg'


def run_photo(api, sources, perceive_res, plan_res, emit):
    for src in sources:
        img = src['image'].read_bytes()
        files = lambda: [('files', (src['image'].name, img, _mime(src['image'])))]       # noqa: E731
        truth_bytes = len(json.dumps(src['truth']))
        base = {'suite': 'photo', 'n_objects': len(src['truth']['objects'])}
        for res in perceive_res:                                      # what does the model see?
            case = {**base, 'group': 'perceive', 'name': f"{src['name']}@{res}"}
            code, body, wall = call(api, 'post', '/perceive', files=files(), data={'resolution': res})
            score = None
            if body.get('scene'):
                try:
                    score = live_score.perception_score(body['scene'], src['truth'])
                except Exception as e:                                # noqa: BLE001
                    body['message'] = f'scoring: {e}'
            exact = bool(score and score['exact'])
            v = {'outcome': 'correct' if exact else 'partial' if score else 'error', 'passed': exact,
                 'reasons': [] if exact else [body.get('message') or 'the scene differs from the true scene']}
            emit(make_row(case, 'photo', code, body, wall, v, len(img), scene=src['name'], req_resolution=res,
                          **{k: score[k] for k in ('recall', 'precision', 'support_acc', 'zones_truth',
                                                   'zones_found', 'n_pred') if score}))
        for task in src['tasks']:                                     # same task: JSON route, then photo route
            case = {**base, 'group': 'photo_plan', 'name': f"{src['name']}/{task['id']}",
                    'instruction': task['instruction']}
            code, body, wall = call(api, 'post', '/plan', json={'instruction': task['instruction'],
                                                                'scene': src['truth']})
            v = live_score.judge(src['truth'], body, None, task['reference'])
            emit(make_row(case, 'json', code, body, wall, v, truth_bytes, scene=src['name'], task=task['id']))
            for res in plan_res:
                code, body, wall = call(api, 'post', '/plan/upload', files=files(),
                                        data={'instruction': task['instruction'], 'resolution': res})
                score, plan = None, body.get('plan')
                if body.get('scene'):
                    try:
                        score = live_score.perception_score(body['scene'], src['truth'])
                        plan = live_score.translate_plan(plan, score['item_map'], score['place_map'])
                    except Exception:                                 # noqa: BLE001
                        score = None
                v = live_score.judge(src['truth'], body, None, task['reference'], plan=plan)
                emit(make_row(case, 'photo', code, body, wall, v, len(img), scene=src['name'], task=task['id'],
                              req_resolution=res, **{k: score[k] for k in ('recall', 'precision', 'support_acc')
                                                     if score}))


# --------------------------------------------------------------------------- summary
def summarize(rows: list[dict]) -> list[dict]:
    """One line per (suite, group, route): count, share correct, median seconds."""
    out, keys = [], []
    for r in rows:
        k = (r['suite'], r['group'], r['route'])
        if k not in keys:
            keys.append(k)
    for k in keys:
        rs = [r for r in rows if (r['suite'], r['group'], r['route']) == k]
        walls = [r['wall_s'] for r in rs if r['wall_s'] is not None]
        out.append({'suite': k[0], 'group': k[1], 'route': k[2], 'n': len(rs),
                    'passed': sum(r['passed'] for r in rs),
                    'median_s': round(statistics.median(walls), 3) if walls else None,
                    'max_s': round(max(walls), 3) if walls else None})
    return out


def write_results(out: Path, meta: dict, rows: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps({'meta': meta, 'summary': summarize(rows), 'rows': rows},
                                                 indent=1, default=str))
    cols = ['suite', 'group', 'name', 'route', 'n_objects', 'status', 'outcome', 'passed', 'wall_s',
            'perceive_s', 'goal_s', 'plan_s', 'total_s', 'moves', 'optimal', 'planner', 'attempts',
            'sent_bytes', 'resolution', 'levels_tried', 'approx_tokens', 'goal_text', 'message']
    with open(out / 'rows.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    print(f"\n{'suite':7s} {'group':14s} {'route':6s} {'n':>3s} {'correct':>8s} {'median s':>9s} {'max s':>8s}")
    for s in summarize(rows):
        print(f"{s['suite']:7s} {s['group']:14s} {s['route']:6s} {s['n']:3d} "
              f"{s['passed']:>4d}/{s['n']:<3d} {s['median_s'] if s['median_s'] is not None else '-':>9} "
              f"{s['max_s'] if s['max_s'] is not None else '-':>8}")
    wrong = [r for r in rows if r['outcome'] == 'wrong plan']
    if wrong:
        print(f"\n{len(wrong)} plan(s) came back 'ok' but do not do the task (the dangerous kind):")
        for r in wrong[:8]:
            print(f"  {r['suite']}/{r['group']}/{r['name']}: {'; '.join(r['reasons'])[:110]}")


# --------------------------------------------------------------------------- main
def run(api, args, out: Path | None = None) -> tuple[dict, list[dict]]:
    code, health, _ = call(api, 'get', '/health')
    if code != 200:
        raise SystemExit(f'cannot reach the server ({health.get("message")}).\n'
                         '  on HEKATE: ./serve.sh status; from another machine check SERVER.md (port forwarding).')
    mcode, mbody, _ = call(api, 'get', '/memory')
    memory = mbody if mcode == 200 else None
    if memory is not None and args.clear_memory:
        call(api, 'post', '/memory/clear')
        print('experience database cleared')
    elif memory is not None:
        print('note: the experience database is ON: repeated inputs are answered from it and look fast.\n'
              '      start the server with --no-memory for clean timings, or pass --clear-memory.')
    print(f'server: {health}')

    suites = [s.strip() for s in args.suites.split(',') if s.strip()]
    plan: list[tuple[str, list]] = []
    only = args.only
    keep = lambda c: not only or only in f"{c['suite']}/{c['group']}/{c['name']}"           # noqa: E731
    if 'json' in suites:
        plan.append(('json', [c for c in live_cases.json_cases(args.quick) if keep(c)]))
    if 'scale' in suites:
        sizes = [int(x) for x in args.scale.split(',')] if args.scale else ([4, 8, 16] if args.quick else [4, 8, 16, 24, 32, 48])
        plan.append(('scale', [c for c in live_cases.scale_cases(sizes) if keep(c)]))
    if 'photo' in suites:
        srcs = live_cases.photo_sources(args.photos, args.photo_limit or (3 if args.quick else None))
        plan.append(('photo', [s for s in srcs if not only or only in f"photo/{s['name']}"]))
    total = sum(len(v) for _, v in plan)
    if not total:
        raise SystemExit('nothing to run (check --suites / --only / photos)')

    rows: list[dict] = []
    done = [0]

    def emit(row):
        rows.append(row)
        done[0] += 1
        mark = 'ok ' if row['passed'] else '-- '
        print(f"[{done[0]:3d}] {mark}{row['suite']}/{row['group']}/{row['name']:28s} {row['status'] or '-':16s} "
              f"{row['outcome']:13s} {row['wall_s']:7.2f}s", flush=True)

    if not args.no_warmup and plan and plan[0][0] == 'json' and plan[0][1]:
        c = plan[0][1][0]
        print('warm-up request (not recorded) ...')
        call(api, 'post', '/plan', json={'instruction': c['instruction'], 'scene': c['scene']})
    started = time.time()
    for _ in range(max(1, args.repeats)):
        for name, items in plan:
            if name == 'json':
                run_json(api, items, emit)
            elif name == 'scale':
                run_scale(api, items, emit)
            else:
                run_photo(api, items, args.photo_resolutions.split(','), args.plan_resolutions.split(','), emit)
    meta = {'label': args.label, 'started': dt.datetime.fromtimestamp(started).isoformat(timespec='seconds'),
            'seconds': round(time.time() - started, 1), 'server': args.url, 'health': health,
            'memory': memory, 'suites': suites, 'quick': args.quick, 'repeats': args.repeats,
            'photo_resolutions': args.photo_resolutions, 'plan_resolutions': args.plan_resolutions}
    print_summary(rows)
    if out is not None:
        write_results(out, meta, rows)
        print(f'\nresults: {out}/results.json')
    return meta, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url', default='http://127.0.0.1:8010')
    ap.add_argument('--label', default='', help='name of this run, e.g. the model; used in the folder name')
    ap.add_argument('--suites', default='json,scale,photo')
    ap.add_argument('--quick', action='store_true', help='a small subset of every suite')
    ap.add_argument('--only', default='', help='run only cases whose "suite/group/name" contains this text')
    ap.add_argument('--repeats', type=int, default=1, help='repeat everything (timing spread; use --no-memory)')
    ap.add_argument('--scale', default='', help='scene sizes for the scale suite, e.g. 4,8,16,32')
    ap.add_argument('--photos', nargs='*', default=[], help='folders with name.jpg + name.json (true scene)')
    ap.add_argument('--photo-limit', type=int, default=0)
    ap.add_argument('--photo-resolutions', default=RESOLUTIONS, help='levels for /perceive')
    ap.add_argument('--plan-resolutions', default='high,auto', help='levels for /plan from a photo')
    ap.add_argument('--clear-memory', action='store_true')
    ap.add_argument('--no-warmup', action='store_true')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'live'))
    ap.add_argument('--no-report', action='store_true')
    args = ap.parse_args()

    label = args.label or 'run'
    out = Path(args.out) / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{label}"
    run(Http(args.url), args, out)
    if not args.no_report:
        import live_report
        live_report.build([out])


if __name__ == '__main__':
    sys.exit(main())
