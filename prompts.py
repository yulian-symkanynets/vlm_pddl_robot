"""Prompt templates for the three benchmark modes.

pddl : the VLM writes the whole PDDL problem (objects, init, goal) for the fixed domain.
json : the VLM writes only a small scene JSON (support relations + goal); Python generates the PDDL.
plan : baseline M1, the VLM writes the action sequence directly (no planner).
split: two separate calls, like the semantic-communication design:
       1) images -> scene JSON only (no task given)        = laptop side
       2) text only: instruction -> goal (named fields)      = server side
"""
from __future__ import annotations

from scenes import DOMAIN_PATH, Scene, Task, env_description

MODES = ('pddl', 'json', 'plan', 'split')
SPLIT_SEP = '\n##### GOAL CALL #####\n'

SCENE_INTRO = (
    'The image(s) show ONE unchanged tabletop scene{views}. A robot arm with a vacuum gripper '
    'will manipulate the small movable objects. Ignore the robot, cables, emergency stop button, '
    'cable outlets and other equipment. The colored paper squares on the table are target places '
    '(red_zone, yellow_zone, blue_zone, green_zone), not objects.'
)


def object_block(scene: Scene, hints: bool) -> str:
    if hints:
        lines = [f'- {i}: {d["desc"]}' for i, d in scene.items.items()]
        return 'Movable objects (use exactly these IDs):\n' + '\n'.join(lines)
    return ('Find the movable objects yourself. Name each one <color>_<shape> in lowercase, '
            'shape one of: cylinder, block (rectangular prism), cube. Example: red_cylinder.')


def known_facts(scene: Scene) -> str:
    return env_description(scene.env, scene.items)


def domain_text() -> str:
    return DOMAIN_PATH.read_text()


def build_prompt(mode: str, scene: Scene, task: Task, n_views: int, hints: bool = True) -> str:
    views = f' from {n_views} different viewpoints' if n_views > 1 else ''
    intro = SCENE_INTRO.format(views=views)
    objs = object_block(scene, hints)
    if mode == 'pddl':
        return f"""{intro}

{objs}

Write a PDDL PROBLEM file for the domain below. Do not repeat or change the domain.

DOMAIN:
{domain_text()}

KNOWN FACTS about the lab (copy them into :init unchanged, and declare these places and slots in :objects):
{known_facts(scene)}
  (slot-free <slot>) for every slot, because no object is on a paper yet.

TASK: {task.instruction}

Rules for :init (describe only what you see):
- Every object has exactly ONE support: (on-table x), or (on x y) if x rests directly on top of object y, or (in x place).
- Image overlap or touching does not mean one object rests on another.
- (clear x) for every object that has nothing on top of it.
- (hand-empty).
- (localized x) for every object visible well enough to grasp; leave it out for hidden objects.
Rules for :goal: express the TASK with (on x y), (in x place) or (on-table x) facts only.

Return only the problem inside ```pddl ... ```."""

    if mode == 'json':
        return f"""{intro}

{objs}

Places: {' '.join(scene.env['places'])}

TASK: {task.instruction}

Describe the scene and the task goal as JSON. Do not plan any moves.
- "support": for every object, what it rests on DIRECTLY: "table", the ID of the object below it, or a place ID.
  Use "unknown" only if the views really do not show it. Image overlap or touching does not mean support.
- "hidden": IDs of objects that are not visible well enough to grasp (usually empty).
- "goal": list of facts that must be true at the end:
  ["on", "x", "y"]  (x rests on object y), ["in", "x", "<place>"], or ["on-table", "x"].

Return only JSON:
{{"support": {{"<id>": "<support>", ...}}, "hidden": [], "goal": [["in", "<id>", "<place>"], ...]}}"""

    if mode == 'plan':
        return f"""{intro}

{objs}

You control the robot with the actions of this PDDL domain:
{domain_text()}

KNOWN FACTS about the lab:
{known_facts(scene)}
  (slot-free <slot>) for every slot, because no object is on a paper yet.
  The robot starts with an empty gripper and knows the pose of every visible object.

TASK: {task.instruction}

Write the sequence of actions that achieves the task, one action per line in PDDL syntax, e.g.
(pick-from-table red_cylinder)
(place-in red_cylinder red_zone_s1 red_zone)
Every action must satisfy its preconditions in the state at that moment
(for example an object can only be picked if nothing rests on it).
Return only the plan inside ``` ... ```."""

    if mode == 'split':             # call 1: perception only, the task is NOT mentioned
        return f"""{intro}

{objs}

Places: {' '.join(scene.env['places'])}

Describe ONLY the current arrangement. Do not plan anything.
- "support": for every object, what it rests on DIRECTLY right now: "table", the ID of the object
  directly below it, or a place ID. Look carefully for objects standing on top of other objects.
  Use "unknown" only if the views really do not show it.
- "hidden": IDs of objects that are not visible well enough to grasp (usually empty).

Return only JSON:
{{"support": {{"<id>": "<support>", ...}}, "hidden": []}}"""

    raise ValueError(f'unknown mode {mode}')


def build_goal_prompt(scene: Scene, task: Task, hints: bool = True) -> str:
    """Call 2 of split mode: text only, no image, no current state."""
    objs = object_block(scene, hints).replace('Movable objects', 'Objects')
    return f"""A robot will rearrange objects on a table.

{objs}

Places: {' '.join(scene.env['places'])}

Instruction: {task.instruction}

Write the DESIRED FINAL arrangement that the instruction asks for (not the current one, which you do not know).
List only what the instruction requires, one entry per requirement:
  {{"object": "<id>", "on": "<id of the object it must rest on>"}}
  {{"object": "<id>", "in": "<place id>"}}
  {{"object": "<id>", "on": "table"}}
"object" is always the one that is moved or placed on top.

Return only JSON:
{{"goal": [ ... ]}}"""
