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


COLOURS = ('red', 'yellow', 'blue', 'green', 'orange', 'purple', 'violet', 'pink', 'white', 'black',
           'brown', 'grey', 'gray', 'cyan', 'magenta')


SHAPES = ('cube', 'block', 'bar', 'cylinder', 'ball', 'sphere', 'prism', 'cone', 'disc', 'ring', 'plate')


def shape_of(*texts) -> str | None:
    """The first shape word in an explicit shape field, label or id ("blue_cube_1" -> "cube")."""
    for t in texts:
        if t:
            for w in re.findall(r'[a-z]+', str(t).lower()):
                if w in SHAPES:
                    return w
    return None


def colour_of(*texts) -> str | None:
    """The first colour word in an explicit colour field, label or id ("red_zone" -> "red")."""
    for t in texts:
        if t:
            for w in re.findall(r'[a-z]+', str(t).lower()):
                if w in COLOURS:
                    return w
    return None


def normalize_scene(raw: dict) -> dict:
    """Clean a scene abstraction into {items, env, support, hidden, unknown, names, warnings}."""
    if not isinstance(raw, dict):
        raise SceneError('scene must be a JSON object')
    warnings, names = [], Names()
    inline_support: dict = {}                        # unique id -> what the object rests on (as written)
    same_name: dict[str, list[str]] = {}             # id as written -> unique ids (several if repeated)
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
        written = make_id(o.get('id') or o.get('label') or 'obj')
        pid = unique(written)
        same_name.setdefault(written, []).append(pid)
        described = ' '.join(str(o[k]) for k in ('color', 'shape') if o.get(k))   # "red block"
        label = str(o.get('label') or o.get('desc') or described or pid.replace('_', ' '))
        d = {'label': label}
        c = colour_of(o.get('color'), label, pid)
        if c:
            d['color'] = c                          # used to expand "sort by colour"
        sh = shape_of(o.get('shape'), label, pid)
        if sh:
            d['shape'] = sh                         # used by goal formulas
        h = o.get('height_mm')
        if isinstance(h, (int, float)) and h > 0:
            d['height_mm'] = float(h)
        size = next((o[k] for k in ('size_mm', 'width_mm', 'diameter_mm')
                     if isinstance(o.get(k), (int, float)) and o[k] > 0), None)
        if size is not None:
            d['size_mm'] = float(size)              # footprint: a larger object may not go on a smaller one
        items[pid] = d
        names.add(pid, o.get('id'), label)
        sup = next((o[k] for k in ('on', 'on_top_of', 'rests_on', 'in') if isinstance(o.get(k), str)), None)
        if sup is not None:                          # per-object form: {"id": ..., "on": "yellow_block"}
            inline_support[pid] = sup                # filed under the UNIQUE id (two "red_block"s stay apart)
    if not items:
        raise SceneError('scene has no objects')

    # ---- places
    places: dict[str, dict] = {}
    pl = raw.get('places') or raw.get('zones') or raw.get('destinations') or []
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
        c = colour_of(p.get('color'), p.get('label'), pid)
        if c:
            d['color'] = c
        if isinstance(p.get('clearance_mm'), (int, float)):
            d['clearance_mm'] = float(p['clearance_mm'])
        if isinstance(p.get('max_height_mm'), (int, float)) and p['max_height_mm'] > 0:
            d['max_height_mm'] = float(p['max_height_mm'])      # a tower in this place may not be taller
        mo = p.get('max_objects', p.get('max_top'))
        if isinstance(mo, (int, float)) and mo >= 1:
            d['max_objects'] = int(mo)                          # at most this many objects in one tower here
        places[pid] = d
        names.add(pid, p.get('id'), p.get('label'))

    # ---- support
    sup_raw = raw.get('support') or {}
    if isinstance(sup_raw, dict):
        sup_raw = {**inline_support, **sup_raw}      # an explicit support map wins
    for r in raw.get('relations') or []:             # [{"subject": a, "relation": "on", "object": b}]
        if isinstance(r, dict) and str(r.get('relation', r.get('type', 'on'))).lower() in ('on', 'on_top_of'):
            if isinstance(sup_raw, dict):
                sup_raw.setdefault(r.get('subject'), r.get('object'))
    if isinstance(sup_raw, list):                    # [{"object": x, "on": y}] form
        sup_raw = {s.get('object'): s.get('on', s.get('in')) for s in sup_raw if isinstance(s, dict)}
    support, unknown, ambiguous = {}, [], []
    for k, v in sup_raw.items():
        x = k if k in items else names.get(k)
        if x not in items:
            warnings.append(f'support for unknown object {k!r} ignored')
            continue
        if str(v).strip().lower() in ('unknown', 'none', 'null', '?', ''):
            unknown.append(x)
            continue
        twins = [c for c in same_name.get(make_id(str(v)), []) if c != x]
        if len(twins) > 1:                           # "on red_block" but there are several red_blocks
            ambiguous.append((x, v, twins))
            continue
        y = twins[0] if twins else names.get(v)
        if y is None:
            raise SceneError(f'{x} rests on unknown {v!r}')
        support[x] = y
    # Repeated ids: keep only the readings that are physically possible (at most one object on
    # another, no loops). One possible reading: use it. Several: do not guess, ask for a view.
    def loops(x, y):
        seen = {x}
        while y in items:
            if y in seen:
                return True
            seen.add(y)
            y = support.get(y)
        return False
    progress = True
    while ambiguous and progress:
        progress = False
        for entry in list(ambiguous):
            x, v, twins = entry
            carrying = {s for s in support.values() if s in items}
            ok = [c for c in twins if c not in carrying and not loops(x, c)]
            if len(ok) == 1:
                support[x] = ok[0]
                ambiguous.remove(entry)
                progress = True
                warnings.append(f'{x} rests on {v!r}, which names several objects; '
                                f'only {ok[0]} is physically possible')
            elif not ok:
                raise SceneError(f'{x} rests on {v!r}, but no object of that name can carry it')
    for x, v, twins in ambiguous:
        unknown.append(x)
        warnings.append(f'{x} rests on {v!r}, which could be any of {", ".join(twins)}: '
                        f'give every object its own id')
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
    rr = raw.get('rules') if isinstance(raw.get('rules'), dict) else {}
    rules = {'no_larger_on_smaller': bool(rr.get('no_larger_on_smaller',
                                              any('size_mm' in d for d in items.values())))}
    if isinstance(rr.get('max_height_mm'), (int, float)) and rr['max_height_mm'] > 0:
        rules['max_height_mm'] = float(rr['max_height_mm'])     # any tower, also on the table
    if isinstance(rr.get('max_objects'), (int, float)) and rr['max_objects'] >= 1:
        rules['max_objects'] = int(rr['max_objects'])
    limited = [x for x in items if 'height_mm' not in items[x]]
    if limited and (rules.get('max_height_mm') or any('max_height_mm' in d for d in places.values())):
        warnings.append('height limits are set but these objects have no height_mm (counted as 0): '
                        + ', '.join(limited[:8]))
    return {'items': items, 'env': {'places': places}, 'support': support, 'hidden': hidden,
            'unknown': unknown, 'names': names, 'warnings': warnings, 'rules': rules}


def scene_to_json(sc: dict) -> dict:
    """Clean scene back to the wire format (what the server actually used)."""
    return {
        'objects': [{'id': i, **d} for i, d in sc['items'].items()],
        'places': [{'id': p, 'kind': d['kind'], 'open': d['open'], 'capacity': d['slots'],
                    **{k: d[k] for k in ('color', 'clearance_mm', 'max_height_mm', 'max_objects') if k in d}}
                   for p, d in sc['env']['places'].items()],
        'rules': sc.get('rules', {}),
        'support': {**sc['support'], **{u: 'unknown' for u in sc['unknown']}},
        'hidden': sc['hidden'],
    }


# --------------------------------------------------------------------------- goal
def expand_sort(raw_goal, sc: dict):
    """{"sort_by_color": true} -> one {"object": x, "in": <place of x's colour>} per object.
    Objects the goal names explicitly keep their own entry. -> (goal, problem or None)"""
    entries = [raw_goal] if isinstance(raw_goal, dict) else list(raw_goal or []) \
        if isinstance(raw_goal, list) else None
    if entries is None:
        return raw_goal, None
    sort = any((isinstance(g, dict) and g.get('sort_by_color')) or g == 'sort_by_color' for g in entries)
    if not sort:
        return raw_goal, None
    rest = [g for g in entries if not ((isinstance(g, dict) and g.get('sort_by_color')) or g == 'sort_by_color')]
    named = {sc['names'].get(str(g.get('object'))) for g in rest if isinstance(g, dict)}
    by_colour = {}
    for p, d in sc['env']['places'].items():
        if d.get('color'):
            by_colour.setdefault(d['color'], p)
    added = [{'object': x, 'in': by_colour[d['color']]} for x, d in sc['items'].items()
             if x not in named and d.get('color') in by_colour]
    if not added and not rest:
        cols = sorted({d['color'] for d in sc['items'].values() if d.get('color')})
        return None, ('sort by colour: no place has the colour of any object (objects: '
                      + (', '.join(cols) or 'no colours given') + '; places: '
                      + (', '.join(sorted(by_colour)) or 'none with a colour') + ')')
    return rest + added, None


def stack_goal(raw_goal, sc: dict):
    """{"stack_by_color": true} -> (col-ok x) for every object with a colour: one stack per colour,
    order and base chosen by the planner. -> (goal or None if not asked, problem or None)"""
    entries = [raw_goal] if isinstance(raw_goal, dict) else raw_goal if isinstance(raw_goal, list) else []
    flag = [g for g in entries if (isinstance(g, dict) and g.get('stack_by_color')) or g == 'stack_by_color']
    if not flag:
        return None, None
    if len(flag) != len(entries):
        return None, 'stack_by_color cannot be combined with other goal entries yet; send it alone'
    goal = [('col-ok', x) for x, d in sc['items'].items() if d.get('color')]
    if not goal:
        return None, 'stack by colour: no object has a colour (send "color" for each object)'
    return goal, None


def formula_goal(raw_goal, sc: dict):
    """{"formula": {...}} -> [("formula", <json text>)] after checking it against the scene.
    -> (goal or None if not a formula goal, problem or None)"""
    import goal_logic
    entries = [raw_goal] if isinstance(raw_goal, dict) else raw_goal if isinstance(raw_goal, list) else []
    flag = [g for g in entries if isinstance(g, dict) and 'formula' in g]
    if not flag:
        return None, None
    parts = []                                  # every formula entry, and plain entries, must hold together
    for g in entries:
        if isinstance(g, dict) and 'formula' in g:
            f = g['formula']
            parts.append(json.loads(f) if isinstance(f, str) else f)
        else:
            fixed = normalize_goal([g])
            if not fixed:
                return None, ('the goal formula can be combined only with other formulas and plain '
                              f'{{"object", "on"/"in"}} entries, not with {json.dumps(g)[:60]}')
            parts.append(fixed_to_formula(fixed))
    formula = parts[0] if len(parts) == 1 else {'and': parts}
    try:
        goal_logic.compile_goal(formula, sc)
    except (goal_logic.FormulaError, json.JSONDecodeError) as e:
        return None, f'the goal formula is invalid: {e}'
    return [('formula', json.dumps(formula, sort_keys=True))], None


def zone_stack_goal(raw_goal, sc: dict):
    """{"stack_in_zones": true} -> one tower per colour standing in the zone of that colour.
    -> ([("zstack", <json {colour: zone}>)] or None if not asked, problem or None)"""
    entries = [raw_goal] if isinstance(raw_goal, dict) else raw_goal if isinstance(raw_goal, list) else []
    flag = [g for g in entries if (isinstance(g, dict) and g.get('stack_in_zones')) or g == 'stack_in_zones']
    if not flag:
        return None, None
    if len(flag) != len(entries):
        return None, 'stack_in_zones cannot be combined with other goal entries; send it alone'
    zone_of = {}
    for p, d in sc['env']['places'].items():
        if d.get('color'):
            zone_of.setdefault(d['color'], p)
    missing = sorted({d.get('color') or x for x, d in sc['items'].items() if d.get('color') not in zone_of})
    if missing:
        return None, ('stack in zones: no zone for ' + ', '.join(missing) +
                      ' (every object needs a colour and a zone of that colour)')
    used = {d['color'] for d in sc['items'].values()}
    return [('zstack', json.dumps({c: z for c, z in zone_of.items() if c in used}, sort_keys=True))], None


def resolve_goal(raw_goal, sc: dict) -> tuple[list[tuple[str, ...]] | None, str | None]:
    """Normalize + ground the goal on the scene ids. Returns (goal, problem_for_the_user)."""
    zoned, problem = zone_stack_goal(raw_goal, sc)
    if problem or zoned:
        return zoned, problem
    stacked, problem = stack_goal(raw_goal, sc)
    if problem or stacked:
        return stacked, problem
    formula, problem = formula_goal(raw_goal, sc)
    if problem or formula:
        return formula, problem
    raw_goal, problem = expand_sort(raw_goal, sc)
    if problem:
        return None, problem
    if isinstance(raw_goal, list) and not raw_goal:
        return None, 'the model wrote an empty goal for this instruction (no object, no place)'
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


def _norm_words(text: str) -> str:
    return ' '.join(re.findall(r'\w+', str(text).lower()))


def backed_by_words(constraints: dict, evidence, instruction: str) -> tuple[dict, list[str]]:
    """Keep a constraint only when its quoted evidence occurs in the instruction."""
    ev = evidence if isinstance(evidence, dict) else {}
    text = _norm_words(instruction)
    kept, dropped = {}, []
    for k, v in constraints.items():
        if v in (None, False, [], {}, ''):
            continue
        q = _norm_words(ev.get(k, ''))
        if q and q in text:
            kept[k] = v
        else:
            dropped.append(k)
    return kept, dropped


def explain_no_plan(sc: dict, goal) -> str:
    return '; '.join(certain_no_plan(sc, goal)) or 'the planner found no sequence of actions that reaches the goal'


def certain_no_plan(sc: dict, goal) -> list[str]:
    """Reasons that make the goal impossible, found without search (microseconds).
    Checked BEFORE planning: otherwise an impossible goal keeps the optimal planner busy
    until its timeout and the fallback planners until theirs, about a minute in total."""
    items, places = sc['items'], sc['env']['places']
    reasons = []
    import pddl_task
    for g in goal:
        if g[0] == 'on' and g[2] in items and pddl_task._no_top(items[g[2]]):
            reasons.append(f'nothing can stand on {g[2]} (it has no flat top)')
    for p, d in places.items():
        want = [g[1] for g in goal if g[0] == 'in' and g[2] == p]
        if len(want) > d['slots']:
            reasons.append(f'{p} holds {d["slots"]} objects, the goal puts {len(want)} in it')
        c = d.get('clearance_mm')
        for x in want:
            h = items[x].get('height_mm')
            if c is not None and h is not None and h > c:
                reasons.append(f'{x} ({h:g} mm) is too tall for {p} ({c:g} mm clearance)')
    return reasons


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
        colour = f', colour {d["color"]}' if d.get('color') else ''
        pls.append(f'- {p}: {d["kind"]}{colour} ({", ".join(extra)})')
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
  {{"sort_by_color": true}}                     every object goes into the place of its own colour
  {{"stack_by_color": true}}                    one stack per colour on the table, any order (write it alone)
  {{"stack_in_zones": true}}                    one tower per colour standing in the zone of that colour (write it alone)
  {{"formula": F}}                              any other goal, as a logic formula (write it alone), see below
For "make every stack one colour" (or "stack the objects by colour") write only {{"stack_by_color": true}}.
For "sort by colour" (or "put everything where its colour belongs") write the one entry
{{"sort_by_color": true}} instead of one entry per object; add separate entries only for objects that
the instruction sends somewhere else.
"object" is always the one that is moved or placed on top. A stack of three needs two entries
(top on middle, middle on bottom). List only what the instruction requires.
If none of these fits, write ONE entry {{"formula": F}} that says what must be TRUE AT THE END.
F is built from (every node has one key, except forall/exists):
  {{"and": [F, ..]}} {{"or": [F, ..]}} {{"not": F}} {{"imply": [F, F]}}
  {{"forall": {{"x": "object"}}, "where": F, "body": F}}   {{"exists": {{"x": "object", "p": "place"}}, "where": F, "body": F}}
  facts: {{"on": ["x", "y"]}} x directly on y, {{"on_table": ["x"]}}, {{"in": ["x", "p"]}}, {{"clear": ["x"]}} nothing on x,
         {{"above": ["x", "y"]}} somewhere above y in a stack,
         {{"color": ["x", "red"]}}, {{"shape": ["x", "cube"]}}, {{"same_color": ["x", "y"]}}, {{"same_shape": ["x", "y"]}}, {{"is": ["x", "y"]}}
Names are ids from the lists above or variables you declare. Example, "put every cube into the red zone":
{{"formula": {{"forall": {{"x": "object"}}, "where": {{"shape": ["x", "cube"]}}, "body": {{"in": ["x", "red_zone"]}}}}}}
If the instruction gives an ORDER ("first ..., then ...", "after that"), list the entries in that order
and add "ordered": true next to "goal".
If the instruction also limits HOW it may be done, add "constraints" (only what the instruction says),
and for every constraint copy the EXACT words of the instruction that say it into "evidence":
  "no_table": true                       nothing may be put down on the table
  "keep": ["<id>", ...]                  these objects must not be moved
  "never_on": [["<id or *>", "<id, place id or table>"], ...]   never place the first on the second
  "first": ["<id>"]                      this object is moved before anything else
If the instruction mentions something that is not listed, is ambiguous, or cannot be written this way,
return {{"goal": [], "question": "<short question for the user>"}}.

Most instructions have no constraints: then leave "constraints" empty.

Return only JSON:
{{"goal": [ ... ], "ordered": false, "constraints": {{ ... }}, "evidence": {{"<constraint>": "<exact words from the instruction>"}}}}"""


def _json_answer(text: str):
    raw = extract_json(strip_reasoning(text))
    if raw is None:
        return None, 'no JSON in the model answer'
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as e:
        return None, f'bad JSON in the model answer: {e}'


# --------------------------------------------------------------------------- planner
OPTIMAL_TIMEOUT_S = 5.0          # optimal search, only when the fast plan is not proven shortest
SATISFICING_TIMEOUT_S = 20.0     # fast planner; certain impossibilities are caught before planning


FIXED = ('on', 'in', 'on-table')


def needs_generic(sc: dict, cons: dict) -> bool:
    """Limits the PDDL domain cannot express (tower height and size rules): use the general search."""
    r = sc.get('rules') or {}
    return bool(cons.get('max_stack_height') or r.get('max_height_mm') or r.get('max_objects')
                or (r.get('no_larger_on_smaller') and any('size_mm' in d for d in sc['items'].values()))
                or any('max_height_mm' in d or 'max_objects' in d for d in sc['env']['places'].values()))


def fixed_to_formula(goal) -> dict:
    return {'and': [{'on_table': [g[1]]} if g[0] == 'on-table' else {g[0]: [g[1], g[2]]} for g in goal]}


def plan_for(sc: dict, goal, planner: str = 'auto', constraints: dict | None = None, memory=None,
             ordered: bool = False) -> dict:
    """Scene + grounded goal (+ the task's constraints) -> PDDL domain and problem for THIS
    request -> plan -> verified on the same problem."""
    import pddl_task
    cons = constraints or {}
    if ordered and goal and len(goal) > 1 and all(g[0] in FIXED for g in goal):
        return plan_ordered(sc, goal, planner, cons)
    impossible = certain_no_plan(sc, goal)
    if impossible:
        return {'status': 'no_plan', 'message': '; '.join(impossible), 'plan': None,
                'planner': 'precheck', 'planner_status': 'unsolvable', 'plan_s': 0.0}
    if goal and goal[0][0] == 'formula':
        return plan_formula(sc, goal[0][1], cons)
    if goal and goal[0][0] == 'zstack':
        return plan_zone_stacks(sc, goal[0][1], cons)
    if goal and needs_generic(sc, cons):
        import goal_logic
        import plan_quality
        lb = plan_quality.lower_bound(sc['support'], sc['items'], goal)       # still valid under limits
        if all(g[0] == 'col-ok' for g in goal):
            return plan_formula(sc, json.dumps(goal_logic.STACK_BY_COLOUR, sort_keys=True), cons, lower_bound=lb)
        if all(g[0] in FIXED for g in goal):
            return plan_formula(sc, json.dumps(fixed_to_formula(goal), sort_keys=True), cons, lower_bound=lb,
                                must_move=plan_quality.must_move(sc['support'], sc['items'], goal))
    stack = bool(goal) and all(g[0] == 'col-ok' for g in goal)      # one stack per colour
    ptext = make_problem('openworld', sc['env'], sc['items'], sc['support'], goal, sc['hidden'])
    domain_pddl, domain_path = None, None
    custom = pddl_task.needs_custom_domain(cons, sc)
    if custom or stack:
        domain_pddl = pddl_task.make_domain(cons, sc) if custom else DOMAIN_PATH.read_text()
        if custom:
            ptext = pddl_task.apply_to_problem(ptext, cons, sc)
        if stack:
            domain_pddl = pddl_task.stack_by_colour_domain(domain_pddl)
            ptext = pddl_task.stack_by_colour_problem(ptext, sc)
        domain_path = pddl_task.write_domain(domain_pddl)
    try:
        problem = parse_problem(ptext, domain_path) if domain_path else parse_problem(ptext)
        hit, mem_state = None, None
        if memory is not None:                    # the exact same problem was solved before
            pkey = memory.key('plan', ptext, domain_pddl or DOMAIN_PATH.read_text(), planner,
                              OPTIMAL_TIMEOUT_S, SATISFICING_TIMEOUT_S)
            hit = memory.get('plan', pkey)
            mem_state = 'hit' if hit is not None else 'miss'
        if hit is not None:
            t_mem = time.perf_counter()
            acts = [(a, list(args)) for a, args in hit['acts']]
            status, used = hit['status'], hit['planner']
            dt = time.perf_counter() - t_mem
        elif stack and planner == 'auto' and not sc['hidden']:
            cs = colour_stack_plan(sc) if not cons else None
            if cs is None:       # too many objects for the exact search (about 20), or constraints:
                import goal_logic          # the general search handles both and is much faster than PDDL here
                return plan_formula(sc, json.dumps(goal_logic.STACK_BY_COLOUR, sort_keys=True), cons)
            status, acts, dt, used = cs
        else:
            if planner == 'auto':
                # 1. a fast planner (about 0.1 s). 2. if its plan moves only objects that must move,
                # each once, no plan can be shorter: done. 3. otherwise optimal search, and keep the
                # fast plan if that search does not finish in time (the response says so).
                import plan_quality
                status, acts, dt, used = solve(problem, 'auto', timeout=SATISFICING_TIMEOUT_S)
                lb = plan_quality.lower_bound(sc['support'], sc['items'], goal)
                if acts is not None and len(plan_quality.moves_of(acts)) > lb:
                    s2, a2, dt2, u2 = solve(problem, 'fast-downward-opt', timeout=OPTIMAL_TIMEOUT_S)
                    dt += dt2
                    if a2 is not None:
                        status, acts, used = s2, a2, u2
            else:
                status, acts, dt, used = solve(problem, planner)
    finally:
        if domain_path:
            domain_path.unlink(missing_ok=True)
    out = {'problem_pddl': ptext, 'planner': used, 'planner_status': str(status), 'plan_s': round(dt, 3)}
    if domain_pddl:
        out['domain_pddl'] = domain_pddl
    if acts is None:
        return {**out, 'status': 'no_plan', 'message': explain_no_plan(sc, goal), 'plan': None}
    sim = simulate(problem, acts)
    if sim['ok']:
        import plan_quality
        out['quality'] = plan_quality.assess(sc, goal, acts, used, str(status))
        if used == 'colour-stacks' and str(status) == 'solved_optimal':
            out['quality'].update(optimal=True, max_extra_moves=0,
                                  proof='exact search for one stack per colour: A* whose estimate '
                                        '(objects that must still move) is never too high')
        if mem_state == 'miss':
            memory.put('plan', pkey, {'acts': acts, 'planner': used, 'status': str(status)})
    if mem_state:
        out['memory_plan'] = mem_state
    return {**out, 'status': 'ok' if sim['ok'] else 'error',
            'message': None if sim['ok'] else f'plan failed verification: {sim["fail_reason"]}',
            'verified': sim['ok'],
            # helper actions of a per-request domain keep the robot's normal action names
            'plan': robot_plan(acts)}


def verify_acts(sc: dict, acts, cons: dict):
    """Run actions on this scene's PDDL problem (with the task's constraints). -> (simulation, problem)."""
    import pddl_task
    ptext = make_problem('openworld', sc['env'], sc['items'], sc['support'], [('hand-empty',)], sc['hidden'])
    domain_path = None
    if pddl_task.needs_custom_domain(cons, sc):
        domain_path = pddl_task.write_domain(pddl_task.make_domain(cons, sc))
        ptext = pddl_task.apply_to_problem(ptext, cons, sc)
    try:
        sim = simulate(parse_problem(ptext, domain_path) if domain_path else parse_problem(ptext), acts)
    finally:
        if domain_path:
            domain_path.unlink(missing_ok=True)
    return sim, ptext


def plan_formula(sc: dict, formula_json: str, cons: dict, heuristic=None, lower_bound: int | None = None,
                 label: str = 'goal-search', must_move: list | None = None) -> dict:
    """General goal: search for moves (formula_plan.py), then check the plan three times: the domain's
    simulator (every action allowed under the constraints), the scene's limits (sizes, tower height),
    and the formula on the final arrangement. `lower_bound` (moves no plan can avoid) gives a proof
    of shortest when the plan reaches it."""
    import formula_plan
    import goal_logic
    goal = goal_logic.compile_goal(formula_json, sc)
    base = {'planner': label, 'plan_s': 0.0}
    if sc['hidden']:
        return {**base, 'status': 'no_plan', 'plan': None, 'planner_status': 'unsupported',
                'message': 'this goal needs every object visible; hidden: ' + ', '.join(sc['hidden'])}
    found = formula_plan.find_moves(sc, goal, cons, heuristic=heuristic)
    base.update(plan_s=found['seconds'], planner_status='solved_satisficing' if found['moves'] is not None else 'unsolved')
    if found['moves'] is None:
        return {**base, 'status': 'no_plan', 'message': found['reason'], 'plan': None}
    acts = formula_plan.to_actions(found['moves'], sc)
    if cons.get('first') and acts:                           # the domain marks the first pick
        acts[0] = (acts[0][0].replace('pick-', 'pick-first-', 1), acts[0][1])
    sim, ptext = verify_acts(sc, acts, cons)
    broken = formula_plan.check_moves(sc, found['moves'], cons)
    final = dict(sc['support'])
    for x, dst in found['moves']:
        final[x] = dst
    holds = goal.holds(goal_logic.View(final, sc['items']))
    ok = sim['ok'] and holds and broken is None
    n = len(found['moves'])
    picked = [x for x, _ in found['moves']]
    proven = n == 0 or (lower_bound is not None and n == lower_bound)
    quality = {'moves': n, 'lower_bound': lower_bound, 'optimal': proven,
               'proof': 'the goal already holds' if n == 0 else
               'the plan moves only objects that must move, each once' if proven else None,
               'max_extra_moves': n - lower_bound if lower_bound is not None else None,
               'moved_more_than_once': sorted({x for x in picked if picked.count(x) > 1}),
               'note': None if proven else
               f'verified, not proven shortest (first plan {found["first"]} moves, best found {n}, '
               f'{found["expanded"]} arrangements searched)'}
    if must_move is not None:
        quality['must_move'] = sorted(must_move)
    return {**base, 'problem_pddl': ptext, 'quality': quality, 'verified': ok,
            'status': 'ok' if ok else 'error',
            'message': None if ok else ('plan failed verification: ' +
                                        (sim['fail_reason'] or broken or 'goal not reached')),
            'plan': robot_plan(acts)}


def zone_towers_impossible(sc: dict, zone_of: dict, cons: dict) -> list[str]:
    """Reasons the colour towers cannot fit their zones at all, found without search: the whole colour
    must stand in one tower, so its height and its number of objects must fit the limits."""
    rules, out = sc.get('rules') or {}, []
    for c, z in zone_of.items():
        objs = [x for x, d in sc['items'].items() if d.get('color') == c]
        d = sc['env']['places'][z]
        height = sum(sc['items'][x].get('height_mm', 0.0) for x in objs)
        for lim, what in ((d.get('max_height_mm'), f'{z} allows {d.get("max_height_mm"):g} mm'
                           if d.get('max_height_mm') else ''),
                          (rules.get('max_height_mm'), f'towers may be {rules.get("max_height_mm"):g} mm'
                           if rules.get('max_height_mm') else '')):
            if lim and height > lim + 1e-6:
                out.append(f'the {len(objs)} {c} objects make a {height:g} mm tower, {what}')
        for lim in (d.get('max_objects'), rules.get('max_objects'), cons.get('max_stack_height')):
            if lim and len(objs) > lim:
                out.append(f'{len(objs)} {c} objects must stand in one tower in {z}, at most {lim} allowed')
    if cons.get('keep'):
        for x in cons['keep']:
            c = sc['items'].get(x, {}).get('color')
            if c in zone_of and sc['support'].get(x) not in sc['items'] and sc['support'].get(x) != zone_of[c]:
                out.append(f'{x} must stay where it is, but it has to go into {zone_of[c]}')
    return out


def plan_zone_stacks(sc: dict, zone_json: str, cons: dict) -> dict:
    """One tower per colour standing in the zone of that colour (works with no_table and limits)."""
    import colour_stacks
    import goal_logic
    zone_of = json.loads(zone_json)
    colour = {x: d.get('color') for x, d in sc['items'].items()}
    impossible = zone_towers_impossible(sc, zone_of, cons)
    if impossible:
        return {'status': 'no_plan', 'message': '; '.join(impossible), 'plan': None, 'planner': 'precheck',
                'planner_status': 'unsolvable', 'plan_s': 0.0}
    guide = lambda sup: colour_stacks.zone_unsettled(sup, colour, zone_of)       # noqa: E731
    must = colour_stacks.zone_unsettled_set(sc['support'], colour, zone_of)
    return plan_formula(sc, json.dumps(goal_logic.stack_in_zones_formula(zone_of), sort_keys=True), cons,
                        heuristic=guide, lower_bound=len(must), label='zone-towers', must_move=must)


def _fact_holds(f, sup: dict) -> bool:
    return sup.get(f[1]) == ('table' if f[0] == 'on-table' else f[2])


def _fact_words(f) -> str:
    return f'{f[1]} on the table' if f[0] == 'on-table' else f'{f[1]} {f[0]} {f[2]}'


def _apply_steps(sup: dict, steps) -> dict:
    sup = dict(sup)
    for st in steps:
        a, args = st['action'], st['args']
        if a == 'place-on-table':
            sup[args[0]] = 'table'
        elif a == 'place-on-item':
            sup[args[0]] = args[1]
        elif a == 'place-in':
            sup[args[0]] = args[-1]
    return sup


def order_check(start: dict, steps, goal) -> dict:
    """When did each goal become true for good (move number, 0 = true from the start)? In order if
    those moments do not decrease along the goal list."""
    sup, move, became = dict(start), 0, []
    state = [_fact_holds(f, sup) for f in goal]
    became = [0 if v else None for v in state]
    for st in steps:
        if not st['action'].startswith('place'):
            continue
        move += 1
        sup = _apply_steps(sup, [st])
        for i, f in enumerate(goal):
            now = _fact_holds(f, sup)
            if now and not state[i]:
                became[i] = move
            if not now:
                became[i] = None
            state[i] = now
    seq = [b for b in became if b]                 # 0 = true from the start and never undone: no order needed
    return {'achieved_at_move': became, 'in_order': None not in became and seq == sorted(seq)}


def plan_ordered(sc: dict, goal, planner: str, cons: dict) -> dict:
    """Goals in the given order ("first A, then B, then C") as ONE search (formula_plan.find_ordered):
    a reached goal is never undone and a later goal is never met before its turn. Exact A* within a fixed
    budget, so the plan is the shortest and the same for the same request. Fallback (hidden objects,
    budget exhausted): stage by stage."""
    import formula_plan
    import plan_quality
    if sc['hidden']:
        return plan_ordered_staged(sc, goal, planner, cons)
    found = formula_plan.find_ordered(sc, goal, cons)
    if found['moves'] is None:
        if found['stopped'] == 'exhausted':
            return {'status': 'no_plan', 'plan': None, 'planner': 'ordered', 'planner_status': 'unsolvable',
                    'plan_s': found['seconds'],
                    'message': 'the goals cannot be met in this order: every way to reach a later goal '
                               'undoes an earlier one or meets a goal before its turn'}
        return plan_ordered_staged(sc, goal, planner, cons)
    acts = formula_plan.to_actions(found['moves'], sc)
    if cons.get('first') and acts:
        acts[0] = (acts[0][0].replace('pick-', 'pick-first-', 1), acts[0][1])
    sim, ptext = verify_acts(sc, acts, cons)
    broken = formula_plan.check_moves(sc, found['moves'], cons)
    steps = robot_plan(acts)
    order = order_check(sc['support'], steps, goal)
    end = _apply_steps(sc['support'], steps)
    ok = sim['ok'] and broken is None and order['in_order'] and all(_fact_holds(f, end) for f in goal)
    quality = plan_quality.assess(sc, goal, steps, 'ordered', '')
    quality['must_move'] = sorted(plan_quality.must_move(sc['support'], sc['items'], goal))
    if found['optimal']:
        quality.update(optimal=True, max_extra_moves=0,
                       proof='exact search over the ordered task (A*, the estimate never overestimates)')
    elif not quality['proof']:
        quality['note'] = (f'the exact search stopped after {found["expanded"]} arrangements; '
                           'verified, not proven shortest')
    return {'status': 'ok' if ok else 'error', 'planner': 'ordered', 'planner_status':
            'solved_optimal' if found['optimal'] else 'solved_satisficing', 'plan_s': found['seconds'],
            'problem_pddl': ptext, 'quality': quality, 'verified': ok, 'order': order, 'plan': steps,
            'message': None if ok else 'plan failed verification: ' +
            (sim['fail_reason'] or broken or 'goals not met in order')}


def plan_ordered_staged(sc: dict, goal, planner: str, cons: dict) -> dict:
    """Fallback: plan stage by stage. Stage k reaches goals 1..k while the objects of goals 1..k-1 stay
    where they are, so an earlier goal is never undone (not proven shortest)."""
    import plan_quality
    t0 = time.perf_counter()
    cur, steps, keep, hidden = dict(sc['support']), [], set(cons.get('keep') or []), list(sc['hidden'])
    for k, g in enumerate(goal, 1):
        sub = goal[:k]
        if not all(_fact_holds(f, cur) for f in sub):
            c2 = {kk: v for kk, v in cons.items() if kk != 'keep' and not (kk == 'first' and steps)}
            if keep:
                c2['keep'] = sorted(keep)
            p = plan_for({**sc, 'support': dict(cur), 'hidden': hidden}, sub, planner, c2)
            if p['status'] != 'ok':
                why = p.get('message') or 'no plan'
                return {**p, 'plan': None, 'message': why if k == 1 else
                        f'goal {k} ({_fact_words(g)}) cannot be reached without undoing goals 1-{k - 1}: {why}'}
            steps += p['plan']
            cur = _apply_steps(cur, p['plan'])
            hidden = [h for h in hidden if not any(s['action'] == 'observe' and s['args'][0] == h for s in p['plan'])]
        keep.add(g[1])
    acts = [(st['action'], st['args']) for st in steps]
    sim, ptext = verify_acts(sc, acts, {kk: v for kk, v in cons.items() if kk != 'first'})
    order = order_check(sc['support'], steps, goal)
    first = cons.get('first')
    first_ok = not first or not steps or plan_quality.moves_of(steps)[:1] == [first]
    ok = sim['ok'] and order['in_order'] and first_ok and all(_fact_holds(f, cur) for f in goal)
    quality = plan_quality.assess(sc, goal, steps, 'ordered', '')
    if quality['proof']:
        quality['proof'] += ' (and in the requested order)'
    else:
        quality['note'] = 'goals planned one after another in the requested order; verified, not proven shortest'
    return {'status': 'ok' if ok else 'error', 'planner': 'ordered', 'planner_status': 'solved_satisficing',
            'plan_s': round(time.perf_counter() - t0, 3), 'problem_pddl': ptext, 'quality': quality,
            'verified': ok, 'order': order, 'plan': steps,
            'message': None if ok else 'plan failed verification: ' + (sim['fail_reason'] or 'goals not met in order')}


def colour_stack_plan(sc: dict):
    """One stack per colour with the dedicated search (colour_stacks.py). Returns the planner
    tuple (status, acts, seconds, name) with the bookkeeping steps the PDDL problem needs for its
    goal check, or None when the scene is outside its scope (objects in zones, no colour)."""
    import colour_stacks
    colour = {x: d.get('color') for x, d in sc['items'].items()}
    if not all(colour.values()) or len(colour) > 18:      # measured: exact search is quick up to ~16-18 objects
        return None
    t0 = time.perf_counter()
    moves, optimal = colour_stacks.solve(sc['support'], colour)
    if moves is None:
        return None
    acts = colour_stacks.to_actions(moves)
    sup = dict(sc['support'])
    for x, _, dst in moves:
        sup[x] = dst
    for x, y in sup.items():                    # mark every object correct, as the goal asks
        acts.append(('claim-base', [x, pddl_task_colour(colour[x])]) if y == 'table'
                    else ('confirm-on', [x, y, pddl_task_colour(colour[x])]))
    return ('solved_optimal' if optimal else 'solved_satisficing', acts,
            time.perf_counter() - t0, 'colour-stacks')


def pddl_task_colour(c: str) -> str:
    import pddl_task
    return pddl_task.colour_token(c)


def robot_plan(acts) -> list[dict]:
    """Planner actions -> the robot's action names. Helper actions of per-request domains are
    renamed back (pick-first-* -> pick-*, place-on-same -> place-on-item, place-as-base ->
    place-on-table, without the colour argument); bookkeeping steps (claim-base, confirm-on)
    move nothing and are left out."""
    import pddl_task
    out = []
    for a, p in acts:
        if a in pddl_task.STACK_BOOKKEEPING:
            continue
        p = list(p)
        if a in pddl_task.STACK_RENAME:
            a, p = pddl_task.STACK_RENAME[a], p[:-1]          # drop the colour argument
        a = a.replace('pick-first-', 'pick-')
        out.append({'action': a, 'args': p, 'pddl': f'({a} {" ".join(p)})'})
    return out


# --------------------------------------------------------------------------- end to end
def run(instruction: str | None = None, scene: dict | None = None, images=None, vlm=None,
        goal=None, planner: str = 'auto', max_visual_tokens: int = 1024, sample: bool = False,
        constraints: dict | None = None, memory=None, places: list | None = None,
        ordered: bool | None = None) -> dict:
    """The server's job. Give `scene` (semantic path) or `images` (+ vlm). `goal` skips the goal call.
    `ordered`: the goal list is a sequence (first A, then B ...); None = as the model read it."""
    t0 = time.perf_counter()
    res = {'status': 'error', 'message': None, 'instruction': instruction, 'timings': {}, 'model_answers': {}}
    pending = []                                  # experience_db entries, stored only if the request ends ok
    if memory is not None:
        res['memory'] = {}

    def done(status, message=None, **kw):
        if status == 'ok' and memory is not None:
            for kind, key, value, meta in pending:
                memory.put(kind, key, value, meta)
        res.update(status=status, message=message, **kw)
        res['timings']['total_s'] = round(time.perf_counter() - t0, 3)
        return res

    # 1) perception (only when no abstraction was sent)
    if scene is None:
        if not images:
            return done('error', 'send "scene" (JSON) or image(s)')
        prompt, g = perceive_prompt(len(images)), None
        if memory is not None:                    # same model, prompt, budget and image bytes -> same answer
            from pathlib import Path
            skey = memory.key('scene', prompt, max_visual_tokens, *[Path(p).read_bytes() for p in images])
            hit = None if sample else memory.get('scene', skey)
            if hit is not None:
                g = {'text': hit, 'generate_s': 0.0}
            res['memory']['scene'] = 'hit' if hit is not None else 'miss'
        if g is None:
            if vlm is None:
                return done('error', 'no model loaded, images cannot be processed')
            g = vlm.generate(list(images), prompt, max_visual_tokens, sample=sample,
                             context={'mode': 'ow_perceive'})
            if memory is not None:
                pending.append(('scene', skey, g['text'], {}))
        res['model_answers']['perceive'] = g['text']
        res['timings']['perceive_s'] = g['generate_s']
        scene, err = _json_answer(g['text'])
        if err:
            return done('error', f'perception: {err}')
    if places:                                    # fixed places of the lab (e.g. coloured papers outside
        scene = dict(scene)                       # the picture): add the ones the camera did not report
        seen = {make_id(p.get('id') if isinstance(p, dict) else p)
                for p in (scene.get('places') or scene.get('zones') or [])}
        extra = [p for p in places if make_id(p.get('id') if isinstance(p, dict) else p) not in seen]
        scene['places'] = list(scene.get('places') or scene.get('zones') or []) + extra
        scene.pop('zones', None)
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
        gp, g = goal_prompt(sc, instruction), None
        if memory is not None:
            ex = memory.examples_block(instruction) if memory.examples else ''
            if ex:                                # similar earlier instructions as examples (optional)
                gp = gp.replace(f'Instruction: {instruction}', ex + f'Instruction: {instruction}', 1)
                res['memory']['examples'] = ex.count('\n- objects:')
            gkey = memory.key('goal', gp)         # the prompt has no arrangement: same objects + words -> same goal
            hit = None if sample else memory.get('goal', gkey)
            if hit is not None:
                g = {'text': hit, 'generate_s': 0.0}
            res['memory']['goal'] = 'hit' if hit is not None else 'miss'
        if g is None:
            if vlm is None:
                return done('error', 'no model loaded, the instruction cannot be interpreted')
            g = vlm.generate([], gp, sample=sample, context={'mode': 'ow_goal', 'scene': sc,
                                                              'instruction': instruction})
            if memory is not None:
                objs = ', '.join(f"{i}: {d['label']}" for i, d in sc['items'].items())
                pending.append(('goal', gkey, g['text'], {'instruction': instruction, 'objects': objs}))
        res['model_answers']['goal'] = g['text']
        res['timings']['goal_s'] = g['generate_s']
        js, err = _json_answer(g['text'])
        if err:
            return done('error', f'goal: {err}')
        if isinstance(js, dict) and js.get('question') and not js.get('goal'):
            return done('clarify', js['question'])
        if constraints is None and isinstance(js, dict) and isinstance(js.get('constraints'), dict):
            # what the sentence forbids, as the model read it; a constraint is kept only if the
            # model quotes words that really are in the instruction (works in any language)
            constraints, dropped = backed_by_words(js['constraints'], js.get('evidence'), instruction)
            if dropped:
                res.setdefault('warnings', []).append(
                    'ignored constraints the instruction does not state: ' + ', '.join(dropped))
        if ordered is None and isinstance(js, dict) and js.get('ordered') is True:
            ordered = True                       # the model read "first ..., then ..." in the sentence
        goal = js.get('goal') if isinstance(js, dict) else js
    grounded, problem = resolve_goal(goal, sc)
    if problem:
        return done('clarify', problem, goal_raw=goal)
    res['ordered'] = bool(ordered)
    if grounded and grounded[0][0] == 'zstack':
        res['goal'] = [{'stack_in_zones': True}]
    elif grounded and grounded[0][0] == 'formula':
        import goal_logic
        formula = json.loads(grounded[0][1])
        res['goal'] = [{'formula': formula, 'in_words': goal_logic.describe(formula)}]
    elif grounded and all(g[0] == 'col-ok' for g in grounded):
        res['goal'] = [{'stack_by_color': True}]             # replanning expands it again on the new scene
    else:
        res['goal'] = [{'object': g[1], 'on': 'table'} if g[0] == 'on-table'
                       else {'object': g[1], g[0]: g[2]} for g in grounded]
    import pddl_task
    cons, problems = pddl_task.normalize_constraints(constraints, sc)
    if problems:
        return done('clarify', '; '.join(problems), goal_raw=goal)
    res['constraints'] = dict(cons)             # max_stack_height is handled by the general search

    # 3) PDDL + planner + verification
    try:
        p = plan_for(sc, grounded, planner, cons, memory=memory, ordered=bool(ordered))
    except Exception as e:                       # noqa: BLE001  (report, do not crash the server)
        return done('error', f'planning: {type(e).__name__}: {e}')
    res['timings']['plan_s'] = p.pop('plan_s')
    if 'memory_plan' in p:
        res['memory']['plan'] = p.pop('memory_plan')
    status, msg = p.pop('status'), p.pop('message')
    res.update(p)
    return done(status, msg)


# --------------------------------------------------------------------------- perception only
def perceive(images, vlm, max_visual_tokens: int = 1024) -> tuple[dict | None, str | None, dict]:
    """Image(s) -> scene JSON written by the VLM (no goal, no planning)."""
    g = vlm.generate(list(images), perceive_prompt(len(images)), max_visual_tokens,
                     context={'mode': 'ow_perceive'})
    scene, err = _json_answer(g['text'])
    return scene, err, g


# --------------------------------------------------------------------------- scene changed
def _after_actions(support: dict, plan: list[dict]) -> list[dict]:
    """The arrangement after 0, 1, 2, ... actions of a plan (only complete pick+place pairs)."""
    cur, states = dict(support), [(0, dict(support))]
    for i, a in enumerate(plan, 1):
        name, args = a['action'], a['args']
        if name == 'place-on-table':
            cur[args[0]] = 'table'
        elif name == 'place-on-item':
            cur[args[0]] = args[1]
        elif name == 'place-in':
            cur[args[0]] = args[2]
        else:
            continue
        states.append((i, dict(cur)))
    return states


def _parse_step(f, plan: list) -> tuple[str | None, list]:
    """A reported step: "(pick-from-item a b)", {"action", "args"}, {"pddl": ...} or {"step": k} (1-based)."""
    if isinstance(f, dict) and isinstance(f.get('step'), int) and 1 <= f['step'] <= len(plan):
        f = plan[f['step'] - 1]
    if isinstance(f, dict) and isinstance(f.get('pddl'), str) and not f.get('action'):
        f = f['pddl']
    if isinstance(f, str):
        t = f.strip().strip('()').split()
        return (t[0], t[1:]) if t else (None, [])
    if isinstance(f, dict) and f.get('action'):
        return str(f['action']), [str(a) for a in f.get('args') or []]
    return None, []


def failed_constraints(failed: list, plan: list, cons: dict) -> tuple[dict, list[str]]:
    """Steps the robot could not do become constraints, so they are never proposed again:
    a failed pick -> the object stays where it is (keep); a failed place -> never put it there."""
    keep = list(cons.get('keep') or [])
    never = [list(t) for t in cons.get('never_on') or []]
    notes = []
    for f in failed or []:
        a, args = _parse_step(f, plan)
        if not a or not args:
            notes.append(f'could not read the failed step {f!r}')
            continue
        x = args[0]
        a = a.replace('pick-first-', 'pick-')
        if a.startswith('pick'):
            if x not in keep:
                keep.append(x)
            notes.append(f'{x} will not be picked again (the pick failed)')
            continue
        target = {'place-on-table': 'table', 'place-on-item': args[1] if len(args) > 1 else None,
                  'place-in': args[-1]}.get(a)
        if target is None:
            notes.append(f'cannot use the failed step {a} {" ".join(args)}')
            continue
        if [x, target] not in never:
            never.append([x, target])
        notes.append(f'{x} will not be put on {target} again (that placement failed)')
    out = dict(cons)
    if keep:
        out['keep'] = keep
    if never:
        out['never_on'] = never
    return out, notes


def replan(previous: dict, new_scene: dict, planner: str = 'auto', memory=None, failed: list | None = None) -> dict:
    """The scene changed after a plan was made (robot progress, a person moved something, an
    object appeared or vanished). Reuses the goal and constraints already understood, so the
    VLM is not called again.

    decision "kept":      the new scene is a state the old plan passes through -> the rest of it
    decision "replanned": anything else -> a new plan from the new scene
    """
    t0 = time.perf_counter()
    goal = previous.get('goal')
    if not goal:
        return {'status': 'error', 'message': 'the previous request has no understood goal to replan for'}
    cons = dict(previous.get('constraints') or {})
    old_support = dict((previous.get('scene') or {}).get('support') or {})
    try:
        sc = normalize_scene(new_scene)
    except SceneError as e:
        return {'status': 'error', 'message': f'scene: {e}'}
    new_support = {x: y for x, y in sc['support'].items()}
    changes = []
    for x in sorted(set(old_support) | set(new_support)):
        a, b = old_support.get(x), new_support.get(x)
        if a is None:
            changes.append(f'{x} appeared on {b}')
        elif b is None:
            changes.append(f'{x} disappeared')
        elif a != b:
            changes.append(f'{x}: {a} -> {b}')
    if cons.get('first') and old_support.get(cons['first']) != new_support.get(cons['first']):
        cons.pop('first')                        # "move X first" is done once X has moved
    old_plan = previous.get('plan') or []
    notes = []
    if failed:                                    # never propose a step the robot could not do
        cons, notes = failed_constraints(failed, old_plan, cons)
    if not failed and set(old_support) == set(new_support) and not sc['unknown']:
        for k, state in _after_actions(old_support, old_plan):
            if state == new_support:
                rest = old_plan[k:]
                import plan_quality
                grounded, _ = resolve_goal(goal, sc)
                quality = plan_quality.assess(sc, grounded or [], rest, '', '')
                if (previous.get('quality') or {}).get('optimal') and not quality['optimal']:
                    # the rest of a shortest plan, from a state on that plan, is itself shortest
                    quality.update(optimal=True, max_extra_moves=0,
                                   proof='remaining part of a plan that was proven shortest')
                return {'status': 'ok', 'decision': 'kept', 'message': None, 'quality': quality,
                        'changes': changes, 'goal': goal, 'constraints': cons,
                        'scene': scene_to_json(sc), 'plan': rest, 'done_actions': k,
                        'timings': {'total_s': round(time.perf_counter() - t0, 3)}}
    res = run(scene=new_scene, goal=goal, constraints=cons, planner=planner, memory=memory,
              ordered=bool(previous.get('ordered')))
    res['decision'] = 'replanned'
    res['changes'] = changes
    if notes:
        res['failed_rules'] = notes
        if res['status'] != 'ok':
            res['message'] = f"{res.get('message')} (excluded after failures: {'; '.join(notes)})"
    return res


def retryable(res: dict) -> bool:
    """A model formatting slip (not a real answer like clarify / no_plan / need_observation)."""
    if not res['model_answers']:
        return False                                  # nothing came from the model
    msg = res.get('message') or ''
    return res['status'] == 'error' or msg.startswith(('goal has an unexpected format',
                                                       'the model wrote an empty goal',
                                                       'the goal formula is invalid'))


def run_until_ok(attempts: int = 3, **kw) -> dict:
    """First try greedy; if the model wrote something unusable, ask again with sampling."""
    history = []
    for k in range(max(1, attempts)):
        res = run(**kw, sample=k > 0)
        history.append({'attempt': k + 1, 'status': res['status'], 'message': res['message'],
                        'total_s': res['timings'].get('total_s')})
        if not retryable(res):
            break
    res['attempts'] = history
    return res


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
        low = instruction.lower()
        if re.search(r'\b(stacks?|towers?)\b', low) and re.search(r'\bzones?\b', low):
            return {'goal': [{'stack_in_zones': True}]}
        if re.search(r'\bfirst\b.*\bthen\b', low):          # an order, not the "first" constraint
            out = self._goal(re.sub(r'\bfirst\b\s*', '', instruction, count=1, flags=re.I), sc)
            return {**out, 'ordered': True} if out.get('goal') else out
        if re.search(r'\bstacks?\b', low) and re.search(r'colou?r', low):
            return {'goal': [{'stack_by_color': True}]}
        if re.search(r'\bsort\b|\bby colou?r\b', low):
            return {'goal': [{'sort_by_color': True}]}
        goal = []
        for part in re.split(r'\s*(?:,|;|\.\s|\bbut\b|\band\b|\bthen\b)\s*', instruction.lower()):
            if re.search(r"\b(do not|don't|never|without|first|leave|keep)\b", part):
                continue                                # a constraint, read below
            m = re.match(r'(?:please\s+)?(?:(?:put|place|move|stack|set)\s+)?(.+?)\s+'
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
        cons, low = {}, instruction.lower()
        if re.search(r"(do not|don't|never|without|no)\b[^.]*\b(table)\b", low):
            cons['no_table'] = True
        m = re.search(r"first (?:move |take |pick )?(?:up )?(?:the )?([a-z_ ]+?)(?:[.,]|$)", low)
        if m and self._find(m.group(1), sc):
            cons['first'] = [self._find(m.group(1), sc)]
        m = re.search(r"(?:leave|keep) (?:the )?([a-z_ ]+?) (?:where it is|in place|alone)", low)
        if m and self._find(m.group(1), sc):
            cons['keep'] = [self._find(m.group(1), sc)]
        ev = {}
        for k, pat in (('no_table', r"(?:do not|don't|never|without|no)\b[^.,]*\btable\b"),
                       ('first', r"first [a-z_ ]+"), ('keep', r"(?:leave|keep) [a-z_ ]+")):
            m = re.search(pat, low)
            if k in cons and m:
                ev[k] = m.group(0)
        return {'goal': goal, **({'constraints': cons, 'evidence': ev} if cons else {})}


def domain_text() -> str:
    return DOMAIN_PATH.read_text()
