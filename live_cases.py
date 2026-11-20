"""Test cases for the live benchmark (live_bench.py).

A case is a plain dict:
    name, group, instruction, scene (what is sent), truth (what is really there; default = scene),
    expect (statuses that count as right; default ["ok"]), reference (formula that must hold at the
    end, written by a person, never shown to the model), check ("_check" block of an example request),
    goal (send the goal directly: no model call), n_objects

Groups (JSON route, the model reads only the instruction):
    examples      examples/requests/*.json that carry a "scene" (wording, towers, no-table ...)
    constraints   the same files that test constraints (first / keep / never_on / no table)
    sort / stack  one sentence at growing scene sizes
    novel         tasks nobody wrote code for: the model has to invent the goal formula
    language      the same kinds of task in Slovak, Czech and German
    refusal       tasks that cannot or must not be done: the right answer is a question or no_plan
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import goal_logic
import openworld

ROOT = Path(__file__).resolve().parent
COLOURS = ['red', 'yellow', 'blue', 'green']
SHAPES = ['cube', 'block', 'bar', 'cylinder']
SORT = 'Sort the objects by colour into the zones.'
STACK = 'Rearrange the objects so that every stack is one colour.'


# --------------------------------------------------------------------------- scenes
def towers_scene(n: int, seed: int = 1, zones: bool = True) -> dict:
    """n objects (4 colours, 4 shapes) in n/4 towers, shuffled; one zone per colour with room for n/4."""
    rnd = random.Random(seed)
    objs = [{'id': f'{COLOURS[i % 4]}_{SHAPES[(i // 4) % 4]}_{i}', 'color': COLOURS[i % 4],
             'shape': SHAPES[(i // 4) % 4]} for i in range(n)]
    rnd.shuffle(objs)
    towers, last = max(1, n // 4), {}
    for k, o in enumerate(objs):
        o['on'] = last.get(k % towers, 'table')
        last[k % towers] = o['id']
    sc = {'objects': objs}
    if zones:
        sc['zones'] = [{'id': f'{c}_zone', 'color': c, 'capacity': max(1, n // 4)} for c in COLOURS]
    return sc


def four_towers() -> dict:
    """The 16 objects of the photo of 20 November (examples/requests/15_four_towers_sort.json)."""
    return json.loads((ROOT / 'examples' / 'requests' / '15_four_towers_sort.json').read_text())['scene']


# --------------------------------------------------------------------------- references
def ref_sort(scene: dict) -> dict:
    """Every object in the zone of its own colour."""
    sc = openworld.normalize_scene(scene)
    zone = {}
    for p, d in sc['env']['places'].items():
        c = d.get('color') or openworld.colour_of(p)
        if c:
            zone.setdefault(c, p)
    return {'and': [{'in': [x, zone[d['color']]]} for x, d in sc['items'].items()
                    if d.get('color') in zone]}


def ref_goal(goal: list) -> dict:
    """Goal tuples (scenes.yaml) -> reference formula."""
    parts = []
    for g in goal:
        if g[0] == 'on-table':
            parts.append({'on_table': [g[1]]})
        else:
            parts.append({g[0]: [g[1], g[2]]})
    return {'and': parts}


REF_STACK = goal_logic.STACK_BY_COLOUR
REF_CUBES_RED = {'forall': {'x': 'object'}, 'where': {'shape': ['x', 'cube']}, 'body': {'in': ['x', 'red_zone']}}
REF_NO_SAME_TOUCH = {'forall': {'x': 'object', 'y': 'object'},
                     'where': {'and': [{'same_color': ['x', 'y']}, {'not': {'is': ['x', 'y']}}]},
                     'body': {'not': {'on': ['x', 'y']}}}
REF_BLUE_ON_TABLE = {'forall': {'x': 'object'}, 'where': {'color': ['x', 'blue']}, 'body': {'on_table': ['x']}}
REF_CYLINDERS_FREE = {'forall': {'x': 'object'}, 'where': {'shape': ['x', 'cylinder']}, 'body': {'clear': ['x']}}
REF_YELLOW_ZONE = {'forall': {'x': 'object'}, 'where': {'color': ['x', 'yellow']},
                   'body': {'in': ['x', 'yellow_zone']}}
REF_RED_TOWER = {'and': [
    {'forall': {'x': 'object'}, 'where': {'color': ['x', 'red']}, 'body': {'or': [
        {'on_table': ['x']},
        {'exists': {'y': 'object'}, 'where': {'color': ['y', 'red']}, 'body': {'on': ['x', 'y']}}]}},
    {'forall': {'x': 'object', 'y': 'object'},
     'where': {'and': [{'color': ['x', 'red']}, {'color': ['y', 'red']}, {'not': {'is': ['x', 'y']}}]},
     'body': {'not': {'and': [{'on_table': ['x']}, {'on_table': ['y']}]}}}]}
REF_YELLOW_TABLE_GREEN_ZONE = {'and': [
    {'forall': {'x': 'object'}, 'where': {'color': ['x', 'yellow']}, 'body': {'on_table': ['x']}},
    {'forall': {'x': 'object'}, 'where': {'color': ['x', 'green']}, 'body': {'in': ['x', 'green_zone']}}]}


# --------------------------------------------------------------------------- JSON suite
def json_cases(quick: bool = False) -> list[dict]:
    cases = []
    four = four_towers()

    def add(group, name, instruction, scene, **kw):
        cases.append({'suite': 'json', 'group': group, 'name': name, 'instruction': instruction,
                      'scene': scene, 'n_objects': len(scene['objects']), **kw})

    for f in sorted((ROOT / 'examples' / 'requests').glob('*.json')):
        req = json.loads(f.read_text())
        if 'scene' not in req or not req.get('scene', {}).get('objects'):
            continue
        group = 'constraints' if f.stem[:2] in ('03', '04', '05', '06', '07', '08') else 'examples'
        chk = req.get('_check') or {}
        want = chk.get('status', 'ok')
        add(group, f.stem, req.get('instruction', ''), req['scene'], goal=req.get('goal'),
            expect=want if isinstance(want, list) else [want], check=chk, note=req.get('_expected'))

    for n in ((4, 8, 16) if quick else (4, 8, 16, 24, 32)):
        sc = towers_scene(n)
        add('sort', f'sort_{n}', SORT, sc, reference=ref_sort(sc))
        add('stack', f'stack_{n}', STACK, sc, reference=REF_STACK)

    novel = [
        ('cubes_to_red', 'Put every cube into the red zone.', REF_CUBES_RED),
        ('no_same_colour_touching', 'Make sure no two objects of the same colour are directly on top '
                                    'of each other.', REF_NO_SAME_TOUCH),
        ('blue_on_table', 'Every blue object must stand on the table.', REF_BLUE_ON_TABLE),
        ('red_tower', 'Build one single tower out of all the red objects.', REF_RED_TOWER),
        ('free_cylinders', 'Nothing may stand on top of a cylinder.', REF_CYLINDERS_FREE),
        ('yellow_to_zone', 'Put all yellow objects into the yellow zone.', REF_YELLOW_ZONE),
        ('two_groups', 'Leave the yellow objects on the table and put the green ones into the green zone.',
         REF_YELLOW_TABLE_GREEN_ZONE),
        ('move_block', 'Move the red block from the first tower onto the second tower.',
         {'on': ['red_block_1', 'blue_block_2']}),
    ]
    for name, text, ref in (novel[:3] if quick else novel):
        add('novel', name, text, four, reference=ref)

    language = [
        ('sk_sort', 'Roztrieď objekty podľa farby do príslušných zón.', ref_sort(four)),
        ('sk_stack', 'Prerob veže tak, aby každá veža mala len jednu farbu.', REF_STACK),
        ('de_cubes', 'Lege alle Würfel in die rote Zone.', REF_CUBES_RED),
        ('cz_red_tower', 'Postav jednu věž ze všech červených objektů.', REF_RED_TOWER),
    ]
    for name, text, ref in (language[:2] if quick else language):
        add('language', name, text, four, reference=ref)

    refusals = [
        ('too_many_for_zone', 'Put every object into the red zone.', ['no_plan', 'clarify']),
        ('zone_missing', 'Put the blue cube into the purple zone.', ['clarify', 'no_plan']),
        ('no_referent', 'Put it on the other one.', ['clarify']),
        ('nonsense', 'Make the objects dance.', ['clarify', 'no_plan']),
        ('object_missing', 'Put the red cube on the green cylinder.', ['clarify', 'no_plan']),
    ]
    for name, text, expect in (refusals[:2] if quick else refusals):
        add('refusal', name, text, four, expect=expect)
    return cases


# --------------------------------------------------------------------------- scaling suite
def scale_cases(sizes) -> list[dict]:
    """No model: the planner alone, with the goal given. Shows time and plan length against size."""
    cases = []
    for n in sizes:
        sc = towers_scene(n)
        for goal_type, goal, ref in (('sort_by_color', [{'sort_by_color': True}], ref_sort(sc)),
                                     ('stack_by_color', [{'stack_by_color': True}], REF_STACK),
                                     ('stack formula', [{'formula': goal_logic.STACK_BY_COLOUR}], REF_STACK)):
            cases.append({'suite': 'scale', 'group': goal_type, 'name': f'{goal_type}_{n}', 'instruction': '',
                          'scene': sc, 'goal': goal, 'reference': ref, 'n_objects': n})
    return cases


# --------------------------------------------------------------------------- photo suite
def _truth_from_yaml(s) -> dict:
    objs = [{'id': i, 'color': d['color'], 'shape': d['shape'], 'label': d.get('desc'),
             'height_mm': d.get('height_mm'), 'on': s.support[i]} for i, d in s.items.items()]
    places = [{'id': p, 'kind': 'zone', 'capacity': d.get('slots', 1), 'open': d.get('open', True)}
              for p, d in s.env['places'].items()]
    return {'objects': objs, 'places': places, 'hidden': list(s.hidden)}


def photo_sources(extra_dirs=(), limit: int | None = None) -> list[dict]:
    """Photos with a true scene: scenes/scenes.yaml (images/) plus any folder given.

    In an extra folder, `name.jpg` needs `name.json` (the true scene, same format as a request's
    "scene"); `name.tasks.json` is optional: [{"id", "instruction", "reference": formula}].
    Without it the photo gets the sort and the stack-by-colour task (when zones exist).
    """
    import scenes
    out = []
    for s in scenes.load_config().scenes:
        truth = _truth_from_yaml(s)
        tasks = [{'id': t.task_id, 'instruction': t.instruction, 'reference': ref_goal(t.goal)} for t in s.tasks]
        for view, path in s.views.items():
            if Path(path).exists():
                out.append({'name': f'{s.id}/{view}', 'image': Path(path), 'truth': truth, 'tasks': tasks,
                            'note': s.note})
    for d in extra_dirs:
        for img in sorted(Path(d).glob('*')):
            truth_file = img.with_suffix('.json')
            if img.suffix.lower() not in ('.jpg', '.jpeg', '.png') or not truth_file.exists():
                continue
            truth = json.loads(truth_file.read_text())
            tasks_file = img.with_suffix('.tasks.json')
            if tasks_file.exists():
                tasks = json.loads(tasks_file.read_text())
            else:
                tasks = [{'id': 'stack', 'instruction': STACK, 'reference': REF_STACK}]
                if truth.get('zones') or truth.get('places'):
                    tasks.insert(0, {'id': 'sort', 'instruction': SORT, 'reference': ref_sort(truth)})
            out.append({'name': img.stem, 'image': img, 'truth': truth, 'tasks': tasks, 'note': 'your photo'})
    return out[:limit] if limit else out
