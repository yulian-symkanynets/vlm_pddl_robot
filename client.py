"""Test client for server.py.

  python client.py health
  python client.py plan  -i "put the yellow block into the box" --scene examples/scenes/box.json
  python client.py plan  -i "swap the stacks" --image images/scene5_reorder_close.jpg
  python client.py perceive images/scene5_reorder_close.jpg images/scene5_reorder_top.jpg
  python client.py solve --scene examples/scenes/shelf.json --goal '[{"object":"white_cube","in":"low_shelf"}]'
  python client.py demo                      # all cases in examples/cases.yaml
  python client.py demo --only box shelf     # cases whose name contains one of the words

--url defaults to $SERVER_URL or http://127.0.0.1:8010
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent
CASES = ROOT / 'examples' / 'cases.yaml'


def show(res: dict, verbose: bool = False):
    st = res.get('status')
    print(f'status : {st}' + (f'   ({res["message"]})' if res.get('message') else ''))
    if res.get('goal'):
        print('goal   : ' + json.dumps(res['goal']))
    if res.get('plan'):
        print(f'plan   : {len(res["plan"])} steps, verified={res.get("verified")}')
        for k, s in enumerate(res['plan'], 1):
            print(f'  {k:2d}. {s["pddl"]}')
    for w in res.get('warnings') or []:
        print(f'warning: {w}')
    t = res.get('timings') or {}
    if t:
        print('timings: ' + ', '.join(f'{k}={v}' for k, v in t.items()))
    if 'received_bytes' in res:
        print(f'sent   : {res["received_bytes"]} bytes   request_id={res.get("request_id")}')
    if verbose:
        if res.get('scene'):
            print('scene  : ' + json.dumps(res['scene']))
        for k, v in (res.get('model_answers') or {}).items():
            print(f'--- model answer ({k}):\n{v}')
        if res.get('problem_pddl'):
            print('--- PDDL problem:\n' + res['problem_pddl'])


def post_plan(url: str, instruction: str | None, scene=None, images=(), goal=None, timeout=600) -> dict:
    if goal is not None:
        r = requests.post(f'{url}/solve', json={'scene': scene, 'goal': goal}, timeout=timeout)
    elif scene is not None:
        r = requests.post(f'{url}/plan', json={'instruction': instruction, 'scene': scene}, timeout=timeout)
    else:
        files = [('files', (Path(p).name, open(p, 'rb'), 'image/jpeg')) for p in images]
        try:
            r = requests.post(f'{url}/plan/upload', data={'instruction': instruction}, files=files,
                              timeout=timeout)
        finally:
            for _, (_, f, _) in files:
                f.close()
    r.raise_for_status()
    return r.json()


def load_json_arg(s: str):
    p = Path(s)
    return json.loads(p.read_text()) if p.exists() else json.loads(s)


def cmd_demo(url: str, only: list[str], verbose: bool):
    model = requests.get(f'{url}/health', timeout=10).json().get('model')
    cases = yaml.safe_load(CASES.read_text())['cases']
    base = CASES.parent
    rows = []
    for c in cases:
        if only and not any(w.lower() in c['name'].lower() for w in only):
            continue
        if model == 'mock' and c.get('real_model_only'):
            rows.append((c['name'], c['expect'], 'skipped (mock)', '', ''))
            continue
        print(f'\n=== {c["name"]}\n    "{c.get("instruction", "")}"')
        scene = json.loads((base / c['scene']).read_text()) if c.get('scene') else None
        images = [str((base / p).resolve()) for p in c.get('images', [])]
        try:
            res = post_plan(url, c.get('instruction'), scene, images, c.get('goal'))
        except requests.RequestException as e:
            res = {'status': 'error', 'message': str(e)}
        show(res, verbose)
        got = res.get('status')
        rows.append((c['name'], c['expect'], got, 'PASS' if got == c['expect'] else 'DIFF',
                     res.get('timings', {}).get('total_s', '')))
    print(f'\n{"case":40s} {"expect":17s} {"got":17s} {"":5s} total_s')
    for r in rows:
        print(f'{r[0][:40]:40s} {r[1]:17s} {str(r[2]):17s} {r[3]:5s} {r[4]}')
    n = sum(r[3] == 'PASS' for r in rows)
    m = sum(r[3] in ('PASS', 'DIFF') for r in rows)
    print(f'\n{n}/{m} as expected (model: {model})')
    return 0 if n == m else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url', default=os.environ.get('SERVER_URL', 'http://127.0.0.1:8010'))
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('health')
    p = sub.add_parser('plan')
    p.add_argument('-i', '--instruction', required=True)
    p.add_argument('--scene', help='scene JSON file or inline JSON')
    p.add_argument('--image', nargs='+', default=[], help='image file(s) instead of --scene')
    p.add_argument('-v', '--verbose', action='store_true')
    p = sub.add_parser('perceive')
    p.add_argument('images', nargs='+')
    p = sub.add_parser('solve')
    p.add_argument('--scene', required=True)
    p.add_argument('--goal', required=True, help='goal JSON file or inline JSON')
    p.add_argument('-v', '--verbose', action='store_true')
    p = sub.add_parser('demo')
    p.add_argument('--only', nargs='*', default=[])
    p.add_argument('-v', '--verbose', action='store_true')
    p = sub.add_parser('get')
    p.add_argument('request_id')
    a = ap.parse_args()
    url = a.url.rstrip('/')

    try:
        if a.cmd == 'health':
            print(json.dumps(requests.get(f'{url}/health', timeout=10).json(), indent=2))
        elif a.cmd == 'plan':
            if not a.scene and not a.image:
                ap.error('plan needs --scene or --image')
            show(post_plan(url, a.instruction, load_json_arg(a.scene) if a.scene else None, a.image), a.verbose)
        elif a.cmd == 'solve':
            show(post_plan(url, None, load_json_arg(a.scene), goal=load_json_arg(a.goal)), a.verbose)
        elif a.cmd == 'perceive':
            files = [('files', (Path(p).name, open(p, 'rb'), 'image/jpeg')) for p in a.images]
            r = requests.post(f'{url}/perceive', files=files, timeout=600)
            r.raise_for_status()
            res = r.json()
            print(json.dumps(res.get('scene') or res, indent=2))
            print(f'\nstatus={res["status"]}  image bytes={res["received_bytes"]}  '
                  f'scene bytes={res.get("scene_bytes")}  timings={res.get("timings")}')
        elif a.cmd == 'demo':
            sys.exit(cmd_demo(url, a.only, a.verbose))
        elif a.cmd == 'get':
            print(json.dumps(requests.get(f'{url}/requests/{a.request_id}', timeout=10).json(), indent=2))
    except requests.ConnectionError:
        sys.exit(f'cannot reach {url}. Is the server running?  ./serve.sh status')
    except requests.HTTPError as e:
        sys.exit(f'HTTP {e.response.status_code}: {e.response.text[:500]}')


if __name__ == '__main__':
    main()
