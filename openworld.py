"""Open-world pipeline: no predefined scenes or tasks.

Input : a user instruction + EITHER a scene abstraction (JSON from the laptop) OR image(s).
Output: a verified plan, or a reason why not (need_observation / clarify / no_plan / error).

    image(s) --VLM call 1--> scene JSON          (semantic encoder; later runs on the laptop)
    scene JSON + instruction --VLM call 2 (text only)--> goal JSON
    scene JSON + goal --Python--> PDDL problem --planner--> plan --simulator--> verified

Scene JSON (everything except "support" is optional):
{
  "objects": [{"id": "red_cylinder", "label": "red cylinder", "height_mm": 60}],
  "places":  [{"id": "box_1", "kind": "box", "open": false, "capacity": 2},
              {"id": "shelf_1", "kind": "shelf", "capacity": 3, "clearance_mm": 50}],
  "support": {"red_cylinder": "table"},          # "table" | object id | place id | "unknown"
  "hidden":  []                                    # visible but not localized well enough to grasp
}
"""
from __future__ import annotations

import json
import re
import time

from evaluate import normalize_goal
from planning import extract_json, parse_problem, simulate, solve, strip_reasoning
from scenes import DOMAIN_PATH, make_problem

DEFAULT_CAPACITY = {'zone': 3, 'box': 3, 'tray': 3, 'shelf': 3, 'bin': 4, 'plate': 1}
RESERVED = {'table', 'unknown', 'and', 'not', 'define', 'problem', 'domain', 'either', 'object'}
STATUSES = ('ok', 'need_observation', 'clarify', 'no_plan', 'error')


# --------------------------------------------------------------------------- names
def make_id(text: str) -> str:
    s = re.sub(r'[^a-z0-9]+', '_', str(text).strip().lower()).strip('_')
    if not s:
        s = 'obj'
    if s[0].isdigit():
        s = 'o_' + s
    if s in RESERVED:
        s = s + '_1'
    return s


class Names:
    """Maps whatever the VLM/laptop wrote (id, label, 'the red cylinder') to a PDDL id."""

    def __init__(self):
        self.alias: dict[str, str] = {}

    def add(self, pid: str, *aliases: str):
        for a in (pid, *aliases):
            if a:
                for k in (str(a).strip().lower(), make_id(a)):
                    self.alias.setdefault(k, pid)

    def get(self, name) -> str | None:
        if name is None:
            return None
        raw = str(name).strip().lower()
        for k in (raw, make_id(raw), make_id(re.sub(r'^(the|a|an)\s+', '', raw))):
            if k in ('table', 'the_table'):
                return 'table'
            if k in self.alias:
                return self.alias[k]
        return None


# --------------------------------------------------------------------------- scene
class SceneError(ValueError):
    pass


def normalize_scene(raw: dict) -> dict:
    """Clean a scene abstraction into {items, env, support, hidden, unknown, names, warnings}."""
    if not isinstance(raw, dict):
        raise SceneError('scene must be a JSON object')
    warnings, names = [], Names()
    used: set[str] = set()

    def unique(base: str) -> str:
        pid, k = base, 2
        while pid in used:
            pid, k = f'{base}_{k}', k + 1
        used.add(pid)
        return pid

    # ---- objects (if missing, take them from the support keys)
    objs = raw.get('objects')
    if objs is None:
        objs = [{'id': k} for k in (raw.get('support') or {})]
    if isinstance(objs, dict):                      # {"id": {...}} form
        objs = [{'id': k, **(v if isinstance(v, dict) else {})} for k, v in objs.items()]
    items: dict[str, dict] = {}
    for o in objs:
        o = o if isinstance(o, dict) else {'id': o}
        pid = unique(make_id(o.get('id') or o.get('label') or 'obj'))
        label = str(o.get('label') or o.get('desc') or pid.replace('_', ' '))
        d = {'label': label}
        h = o.get('height_mm')
        if isinstance(h, (int, float)) and h > 0:
            d['height_mm'] = float(h)
        items[pid] = d
        names.add(pid, o.get('id'), label)
    if not items:
        raise SceneError('scene has no objects')

    # ---- places
    places: dict[str, dict] = {}
    pl = raw.get('places') or []
    if isinstance(pl, dict):
        pl = [{'id': k, **(v if isinstance(v, dict) else {})} for k, v in pl.items()]
    for p in pl:
        p = p if isinstance(p, dict) else {'id': p}
        if make_id(p.get('id') or '') in ('table', 'table_1'):
            continue                                 # the table is not a place
        kind = make_id(p.get('kind') or 'zone')
        pid = unique(make_id(p.get('id') or p.get('label') or kind))
        cap = p.get('capacity', p.get('slots'))
        cap = int(cap) if isinstance(cap, (int, float)) and cap >= 1 else DEFAULT_CAPACITY.get(kind, 3)
        is_open = p.get('open', True)
        if str(p.get('lid', '')).lower() == 'closed':
            is_open = False
        d = {'kind': kind, 'slots': cap, 'open': bool(is_open)}
        if isinstance(p.get('clearance_mm'), (int, float)):
            d['clearance_mm'] = float(p['clearance_mm'])
        places[pid] = d
        names.add(pid, p.get('id'), p.get('label'))

    # ---- support
    sup_raw = raw.get('support') or {}
    if isinstance(sup_raw, list):                    # [{"object": x, "on": y}] form
        sup_raw = {s.get('object'): s.get('on', s.get('in')) for s in sup_raw if isinstance(s, dict)}
    support, unknown = {}, []
    for k, v in sup_raw.items():
        x = names.get(k)
        if x not in items:
            warnings.append(f'support for unknown object {k!r} ignored')
            continue
        if str(v).strip().lower() in ('unknown', 'none', 'null', '?', ''):
            unknown.append(x)
            continue
        y = names.get(v)
        if y is None:
            raise SceneError(f'{x} rests on unknown {v!r}')
        support[x] = y
    for x in items:
        if x not in support and x not in unknown:
            warnings.append(f'no support given for {x}, assuming table')
            support[x] = 'table'

    # a place must have room for what is already in it
    for p, d in places.items():
        n = sum(1 for s in support.values() if s == p)
        if n > d['slots']:
            warnings.append(f'{p}: capacity raised from {d["slots"]} to {n} (that many are already in it)')
            d['slots'] = n

    hidden = [names.get(h) for h in raw.get('hidden') or []]
    hidden = [h for h in hidden if h in items]
    missing_h = [x for x in items if 'height_mm' not in items[x]]
    if missing_h and any('clearance_mm' in d for d in places.values()):
        warnings.append('no height for ' + ', '.join(missing_h) + ': assumed to fit under every shelf')
    return {'items': items, 'env': {'places': places}, 'support': support, 'hidden': hidden,
            'unknown': unknown, 'names': names, 'warnings': warnings}


def scene_to_json(sc: dict) -> dict:
    """Clean scene back to the wire format (what the server actually used)."""
    return {
        'objects': [{'id': i, **d} for i, d in sc['items'].items()],
        'places': [{'id': p, 'kind': d['kind'], 'open': d['open'], 'capacity': d['slots'],
                    **({'clearance_mm': d['clearance_mm']} if 'clearance_mm' in d else {})}
                   for p, d in sc['env']['places'].items()],
        'support': {**sc['support'], **{u: 'unknown' for u in sc['unknown']}},
        'hidden': sc['hidden'],
    }


# --------------------------------------------------------------------------- goal
def resolve_goal(raw_goal, sc: dict) -> tuple[list[tuple[str, ...]] | None, str | None]:
    """Normalize + ground the goal on the scene ids. Returns (goal, problem_for_the_user)."""
    goal = normalize_goal(raw_goal)
    if goal is None:
        return None, f'goal has an unexpected format: {json.dumps(raw_goal)[:200]}'
    items, places, names = sc['items'], sc['env']['places'], sc['names']
    out, unknown = [], []
    for g in goal:
        args = [names.get(a) for a in g[1:]]
        for a, raw in zip(args, g[1:]):
            if a is None:
                unknown.append(raw)
        if None in args:
            continue
        x = args[0]
        if x not in items:
            return None, f'"{x}" is a place, it cannot be moved'
        if g[0] == 'on-table' or (len(args) > 1 and args[1] == 'table'):
            out.append(('on-table', x))
        elif args[1] in places:
            out.append(('in', x, args[1]))       # "on the shelf" -> in shelf
        else:
            out.append(('on', x, args[1]))       # "in the cup" where cup is an object -> on
    if unknown:
        return None, 'not in the scene: ' + ', '.join(sorted(set(unknown)))
    if not out:
        return None, 'the instruction gives no goal'
    # simple sanity checks before calling the planner
    below = [g[2] for g in out if g[0] == 'on']
    if any(g[0] == 'on' and g[1] == g[2] for g in out):
        return None, 'an object cannot rest on itself'
    dup = sorted({b for b in below if below.count(b) > 1})
    if dup:
        return None, f'more than one object should rest on {", ".join(dup)}'
    target = {g[1]: g for g in out}
    if len(target) < len(out):
        return None, 'one object has two different goal positions'
    on = {g[1]: g[2] for g in out if g[0] == 'on'}
    for x in on:
        seen, cur = {x}, on[x]
        while cur in on:
            if cur in seen:
                return None, f'circular stacking through {x}'
            seen.add(cur)
            cur = on[cur]
    return out, None


def explain_no_plan(sc: dict, goal) -> str:
    items, places = sc['items'], sc['env']['places']
    reasons = []
    for p, d in places.items():
        want = [g[1] for g in goal if g[0] == 'in' and g[2] == p]
        if len(want) > d['slots']:
            reasons.append(f'{p} holds {d["slots"]} objects, the goal puts {len(want)} in it')
        c = d.get('clearance_mm')
        for x in want:
            h = items[x].get('height_mm')
            if c is not None and h is not None and h > c:
                reasons.append(f'{x} ({h:g} mm) is too tall for {p} ({c:g} mm clearance)')
    return '; '.join(reasons) or 'the planner found no sequence of actions that reaches the goal'


# --------------------------------------------------------------------------- prompts
def perceive_prompt(n_views: int = 1) -> str:
    views = f' from {n_views} different viewpoints' if n_views > 1 else ''
    return f"""The image(s) show ONE unchanged tabletop work area of a robot arm with a vacuum gripper{views}.
Ignore the robot itself, cables, buttons, cable outlets and other fixed equipment.

Describe the CURRENT scene as JSON. Do not plan anything.
1. "objects": every small movable object the robot could pick up:
   {{"id": "<color>_<shape>", "label": "<short description>", "height_mm": <estimate or null>}}
   id in lowercase with underscores, unique (add _2, _3 for duplicates). shape: cylinder, block, cube, ball, ...
2. "places": every area where objects can be put: colored paper zones, boxes, trays, bins, shelf levels:
   {{"id": "<color>_<kind>", "kind": "zone|box|tray|bin|shelf", "open": true, "capacity": <how many objects fit>}}
   "open": false only for a box with a closed lid. The table itself is not a place.
3. "support": for EVERY object, what it rests on DIRECTLY right now: "table", the id of the object
   directly below it, or a place id. Look carefully for objects standing on top of other objects.
   Touching or overlapping in the image does not mean one rests on the other.
   Use "unknown" only if the views really do not show it.
4. "hidden": ids of objects that are not visible well enough to grasp (usually empty).

Return only JSON:
{{"objects": [...], "places": [...], "support": {{"<id>": "<support>"}}, "hidden": []}}"""


def goal_prompt(sc: dict, instruction: str) -> str:
    """Text only. The current arrangement is NOT given, so the model cannot copy it as the goal."""
    objs = []
    for i, d in sc['items'].items():
        h = f' ({d["height_mm"]:g} mm tall)' if 'height_mm' in d else ''
        objs.append(f'- {i}: {d["label"]}{h}')
    pls = []
    for p, d in sc['env']['places'].items():
        extra = [] if d['open'] else ['lid closed']
        extra.append(f'room for {d["slots"]}')
        pls.append(f'- {p}: {d["kind"]} ({", ".join(extra)})')
    places = '\n'.join(pls) if pls else '- (none)'
    return f"""A robot arm will rearrange objects on a table. This is what is in the scene:

Objects:
{chr(10).join(objs)}

Places:
{places}

Instruction: {instruction}

Write the DESIRED FINAL arrangement that the instruction asks for. Use only the ids above.
One entry per requirement:
  {{"object": "<id>", "on": "<object id>"}}    the object rests directly on top of another object
  {{"object": "<id>", "in": "<place id>"}}
  {{"object": "<id>", "on": "table"}}
"object" is always the one that is moved or placed on top. A stack of three needs two entries
(top on middle, middle on bottom). List only what the instruction requires.
If the instruction mentions something that is not listed, is ambiguous, or cannot be written this way,
return {{"goal": [], "question": "<short question for the user>"}}.

Return only JSON:
{{"goal": [ ... ]}}"""


def _json_answer(text: str):
    raw = extract_json(strip_reasoning(text))
    if raw is None:
        return None, 'no JSON in the model answer'
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as e:
        return None, f'bad JSON in the model answer: {e}'


# --------------------------------------------------------------------------- planner
def plan_for(sc: dict, goal, planner: str = 'auto') -> dict:
    """Scene + grounded goal -> PDDL problem -> plan -> verified on the same problem."""
    ptext = make_problem('openworld', sc['env'], sc['items'], sc['support'], goal, sc['hidden'])
    problem = parse_problem(ptext)
    status, acts, dt, used = solve(problem, planner)
    out = {'problem_pddl': ptext, 'planner': used, 'planner_status': str(status), 'plan_s': round(dt, 3)}
    if acts is None:
        return {**out, 'status': 'no_plan', 'message': explain_no_plan(sc, goal), 'plan': None}
    sim = simulate(problem, acts)
    return {**out, 'status': 'ok' if sim['ok'] else 'error',
            'message': None if sim['ok'] else f'plan failed verification: {sim["fail_reason"]}',
            'verified': sim['ok'],
            'plan': [{'action': a, 'args': list(p), 'pddl': f'({a} {" ".join(p)})'} for a, p in acts]}


# --------------------------------------------------------------------------- end to end
def run(instruction: str | None = None, scene: dict | None = None, images=None, vlm=None,
        goal=None, planner: str = 'auto', max_visual_tokens: int = 1024) -> dict:
    """The server's job. Give `scene` (semantic path) or `images` (+ vlm). `goal` skips the goal call."""
    t0 = time.perf_counter()
    res = {'status': 'error', 'message': None, 'instruction': instruction, 'timings': {}, 'model_answers': {}}

    def done(status, message=None, **kw):
        res.update(status=status, message=message, **kw)
        res['timings']['total_s'] = round(time.perf_counter() - t0, 3)
        return res

    # 1) perception (only when no abstraction was sent)
    if scene is None:
        if not images:
            return done('error', 'send "scene" (JSON) or image(s)')
        if vlm is None:
            return done('error', 'no model loaded, images cannot be processed')
        g = vlm.generate(list(images), perceive_prompt(len(images)), max_visual_tokens,
                         context={'mode': 'ow_perceive'})
        res['model_answers']['perceive'] = g['text']
        res['timings']['perceive_s'] = g['generate_s']
        scene, err = _json_answer(g['text'])
        if err:
            return done('error', f'perception: {err}')
    try:
        sc = normalize_scene(scene)
    except SceneError as e:
        return done('error', f'scene: {e}')
    res['scene'] = scene_to_json(sc)
    res['warnings'] = sc['warnings']
    if sc['unknown']:
        return done('need_observation', 'support unknown for ' + ', '.join(sc['unknown']) +
                    ': send another view or a crop of these objects', need=sc['unknown'])

    # 2) goal (text-only call, or given directly)
    if goal is None:
        if not instruction:
            return done('error', 'send "instruction" or "goal"')
        if vlm is None:
            return done('error', 'no model loaded, the instruction cannot be interpreted')
        g = vlm.generate([], goal_prompt(sc, instruction), context={'mode': 'ow_goal', 'scene': sc,
                                                                     'instruction': instruction})
        res['model_answers']['goal'] = g['text']
        res['timings']['goal_s'] = g['generate_s']
        js, err = _json_answer(g['text'])
        if err:
            return done('error', f'goal: {err}')
        if isinstance(js, dict) and js.get('question') and not js.get('goal'):
            return done('clarify', js['question'])
        goal = js.get('goal') if isinstance(js, dict) else js
    grounded, problem = resolve_goal(goal, sc)
    if problem:
        return done('clarify', problem, goal_raw=goal)
    res['goal'] = [{'object': g[1], 'on': 'table'} if g[0] == 'on-table'
                   else {'object': g[1], g[0]: g[2]} for g in grounded]

    # 3) PDDL + planner + verification
    try:
        p = plan_for(sc, grounded, planner)
    except Exception as e:                       # noqa: BLE001  (report, do not crash the server)
        return done('error', f'planning: {type(e).__name__}: {e}')
    res['timings']['plan_s'] = p.pop('plan_s')
    status, msg = p.pop('status'), p.pop('message')
    res.update(p)
    return done(status, msg)


# --------------------------------------------------------------------------- mock model
DEMO_SCENE = {
    'objects': [{'id': 'red_cylinder', 'label': 'red cylinder', 'height_mm': 60},
                {'id': 'yellow_block', 'label': 'yellow block', 'height_mm': 40},
                {'id': 'green_block', 'label': 'green block', 'height_mm': 40},
                {'id': 'blue_cylinder', 'label': 'blue cylinder', 'height_mm': 60}],
    'places': [{'id': c + '_zone', 'kind': 'zone', 'capacity': 2} for c in ('red', 'yellow', 'blue', 'green')],
    'support': {'red_cylinder': 'yellow_block', 'yellow_block': 'table',
                'green_block': 'blue_cylinder', 'blue_cylinder': 'table'},
    'hidden': [],
}


class RuleMock:
    """No GPU needed. Perception returns DEMO_SCENE; the goal call understands
    'put/place/move/stack X on|onto|in|into Y' (joined with 'and' / ',' / ';')."""
    key, quant, hf, load_s = 'mock', 'none', 'rule-based mock', 0.0

    def generate(self, image_paths, prompt, max_visual_tokens=1024, max_new_tokens=None,
                 sample=False, context=None) -> dict:
        t0 = time.perf_counter()
        if context['mode'] == 'ow_perceive':
            text = json.dumps(DEMO_SCENE)
        else:
            text = json.dumps(self._goal(context['instruction'], context['scene']))
        return {'text': text, 'generate_s': round(time.perf_counter() - t0, 4),
                'input_tokens': len(prompt) // 4, 'output_tokens': len(text) // 4}

    @staticmethod
    def _find(phrase: str, sc: dict):
        phrase = re.sub(r'^(the|a|an)\s+', '', phrase.strip().lower())
        if phrase in ('table', 'the table'):
            return 'table'
        best = sc['names'].get(phrase)
        if best:
            return best
        words = set(re.findall(r'[a-z0-9]+', phrase))
        cands = list(sc['items']) + list(sc['env']['places'])
        scored = [(len(words & set(c.split('_'))
                       | words & set(re.findall(r'[a-z0-9]+', sc['items'].get(c, {}).get('label', '')))), c)
                  for c in cands]
        scored = [s for s in scored if s[0] > 0]
        if not scored:
            return None
        scored.sort(reverse=True)
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            return None                                 # ambiguous
        return scored[0][1]

    def _goal(self, instruction: str, sc: dict) -> dict:
        goal = []
        for part in re.split(r'\s*(?:,|;|\band\b|\bthen\b)\s*', instruction.lower()):
            m = re.match(r'(?:please\s+)?(?:put|place|move|stack|set)\s+(.+?)\s+'
                         r'(on top of|onto|on|into|inside|in|to)\s+(.+?)\.?$', part.strip())
            if not m:
                continue
            x, rel, y = self._find(m.group(1), sc), m.group(2), self._find(m.group(3), sc)
            if x is None or y is None:
                missing = m.group(1) if x is None else m.group(3)
                return {'goal': [], 'question': f'I do not see "{missing}" in the scene. Which object do you mean?'}
            if y == 'table':
                goal.append({'object': x, 'on': 'table'})
            elif y in sc['env']['places']:
                goal.append({'object': x, 'in': y})
            else:
                goal.append({'object': x, 'on': y})
        if not goal:
            return {'goal': [], 'question': 'The mock model only understands "put X on/in Y". '
                                            'Please rephrase (or use a real model).'}
        return {'goal': goal}


def domain_text() -> str:
    return DOMAIN_PATH.read_text()
