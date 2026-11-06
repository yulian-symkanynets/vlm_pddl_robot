"""Scene config loading and automatic PDDL problem generation.

The domain is fixed (domain/tabletop.pddl). A problem is generated from:
  env (places, slots, open/closed, clearance)  +  items  +  support relations  +  goal facts.
The same generator builds the ground-truth problem and the problem from a VLM's JSON scene.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
DOMAIN_PATH = ROOT / 'domain' / 'tabletop.pddl'
SCENES_PATH = ROOT / 'scenes' / 'scenes.yaml'
IMAGES_DIR = ROOT / 'images'


@dataclass
class Task:
    scene_id: str
    task_id: str
    instruction: str
    goal: list[tuple[str, ...]]          # ground truth goal facts


@dataclass
class Scene:
    id: str
    views: dict[str, Path]
    view_sets: list[list[str]]
    items: dict[str, dict]               # id -> {color, shape, height_mm, desc}
    support: dict[str, str]              # ground truth
    hidden: list[str]
    tasks: list[Task]
    env: dict
    note: str = ''

    def view_set_name(self, views: list[str]) -> str:
        return '+'.join(views)


@dataclass
class Config:
    env: dict
    scenes: list[Scene] = field(default_factory=list)

    def scene(self, scene_id: str) -> Scene:
        for s in self.scenes:
            if s.id == scene_id:
                return s
        raise KeyError(scene_id)


def _word(w) -> str:
    # YAML 1.1 reads a bare `on` as boolean True
    if w is True:
        return 'on'
    return str(w)


def load_config(path: Path = SCENES_PATH) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    env = raw['env']
    item_defs = raw['item_defs']
    cfg = Config(env=env)
    for s in raw['scenes']:
        items = {i: dict(item_defs[i]) for i in s['items']}
        scene_env = {'places': {**env['places'], **s.get('env', {}).get('places', {})}}
        tasks = []
        for t in s['tasks']:
            if t['goal'] == 'sort':
                goal = [('in', i, f"{d['color']}_zone") for i, d in items.items()]
                instruction = t.get('instruction', raw['sort_instruction'])
            else:
                goal = [tuple(_word(w) for w in g) for g in t['goal']]
                instruction = t['instruction']
            tasks.append(Task(s['id'], t['id'], instruction, goal))
        cfg.scenes.append(Scene(
            id=s['id'],
            views={k: IMAGES_DIR / v for k, v in s['views'].items()},
            view_sets=s['view_sets'],
            items=items,
            support=dict(s['support']),
            hidden=list(s.get('hidden', [])),
            tasks=tasks,
            env=scene_env,
            note=s.get('note', ''),
        ))
    return cfg


# ---------------------------------------------------------------- PDDL generation

def slot_names(place: str, n: int) -> list[str]:
    return [f'{place}_s{k + 1}' for k in range(n)]


def env_facts(env: dict, items: dict[str, dict]) -> tuple[list[str], list[str], list[str]]:
    """Return (places, slots, facts) that follow from the fixed lab setup."""
    places, slots, facts = [], [], []
    for p, d in env['places'].items():
        places.append(p)
        ss = slot_names(p, int(d.get('slots', 1)))
        slots += ss
        for s in ss:
            facts.append(f'(slot-of {s} {p})')
        facts.append(f'(open {p})' if d.get('open', True) else f'(lid-closed {p})')
        clearance = d.get('clearance_mm')
        for i, idef in items.items():
            if clearance is None or idef.get('height_mm', 0) <= clearance:
                facts.append(f'(fits {i} {p})')
    return places, slots, facts


def state_facts(env: dict, support: dict[str, str], hidden: list[str] = ()) -> list[str]:
    """Facts from support relations: on/on-table/in/occupies, clear, slot-free, localized."""
    facts = ['(hand-empty)']
    used_slots = {}
    places = env['places']
    for x, sup in support.items():
        if sup == 'table':
            facts.append(f'(on-table {x})')
        elif sup in places:
            free = [s for s in slot_names(sup, int(places[sup].get('slots', 1))) if s not in used_slots]
            if not free:
                raise ValueError(f'place {sup} has no free slot for {x}')
            used_slots[free[0]] = x
            facts += [f'(in {x} {sup})', f'(occupies {x} {free[0]})']
        else:
            facts.append(f'(on {x} {sup})')
    covered = {sup for sup in support.values()}
    for x in support:
        if x not in covered:
            facts.append(f'(clear {x})')
        if x not in hidden:
            facts.append(f'(localized {x})')
    for p, d in places.items():
        for s in slot_names(p, int(d.get('slots', 1))):
            if s not in used_slots:
                facts.append(f'(slot-free {s})')
    return facts


def validate_support(support: dict[str, str], items, places) -> None:
    for x, sup in support.items():
        if x not in items:
            raise ValueError(f'unknown item {x}')
        if sup != 'table' and sup not in items and sup not in places:
            raise ValueError(f'unknown support {x} -> {sup}')
        if sup == x:
            raise ValueError(f'{x} rests on itself')
    on_item = [s for s in support.values() if s in items]
    dup = {s for s in on_item if on_item.count(s) > 1}
    if dup:
        raise ValueError(f'more than one item rests on {sorted(dup)}')
    for x in support:                       # cycle check
        seen, cur = {x}, support[x]
        while cur in support:
            if cur in seen:
                raise ValueError(f'cyclic support through {x}')
            seen.add(cur)
            cur = support[cur]


def make_problem(name: str, env: dict, items: dict[str, dict], support: dict[str, str],
                 goal: list[tuple[str, ...]], hidden: list[str] = ()) -> str:
    validate_support(support, items, env['places'])
    places, slots, efacts = env_facts(env, items)
    sfacts = state_facts(env, support, hidden)
    goal_str = '\n    '.join('(' + ' '.join(g) + ')' for g in goal)
    init = '\n    '.join(sfacts + efacts)
    groups = [(items, 'item'), (places, 'place'), (slots, 'slot')]
    objects = '\n    '.join(f'{" ".join(names)} - {t}' for names, t in groups if names)
    return (
        f'(define (problem {name})\n'
        f'  (:domain tabletop)\n'
        f'  (:objects\n'
        f'    {objects})\n'
        f'  (:init\n    {init})\n'
        f'  (:goal (and\n    {goal_str})))\n'
    )


def gt_problem(scene: Scene, task: Task) -> str:
    return make_problem(f'{scene.id}-{task.task_id}', scene.env, scene.items, scene.support,
                        task.goal, scene.hidden)


def env_description(env: dict, items: dict[str, dict]) -> str:
    """Known facts block given to the VLM (it must not invent places/slots)."""
    places, slots, facts = env_facts(env, items)
    lines = [f'places: {" ".join(places)}', f'slots: {" ".join(slots)}', 'facts:']
    lines += ['  ' + f for f in facts]
    return '\n'.join(lines)
