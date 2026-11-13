"""Scripted model answers for running the LLM strategies without a GPU.

Like MockVLM in vlm.py, these answer from the ground truth and can inject the
mistakes a real model typically makes, so the surrounding logic (validation,
re-prompting, fallback) can be exercised. They are stand-ins, not models.
"""
from __future__ import annotations

import json
import random
import re

from semplan.planner import plan
from semplan.task_planner import plan_task
from semplan.scene import parse_scene


def _scene_from_prompt(prompt: str):
    line = next(l for l in prompt.splitlines() if l.startswith('{"') and '"objects"' in l)
    d = json.loads(line)
    m = re.search(r"^Task JSON: (.*)$", prompt, re.M)
    if m:
        d["task"] = json.loads(m.group(1))
    return parse_scene(d)


def planner_oracle(error_rate: float = 0.0, seed: int = 0):
    """Answers plan prompts correctly, except that with probability `error_rate` it
    drops the first un-stacking move (the classic 'picks a covered object' mistake).
    After a validator rejection it answers correctly."""
    rng = random.Random(seed)

    def respond(prompt: str) -> str:
        scene = _scene_from_prompt(prompt)
        moves = (plan_task(scene) if scene.general else plan(scene)).moves
        corrected = "rejected by the validator" in prompt
        if not corrected and rng.random() < error_rate and len(moves) > 1:
            moves = moves[1:]
        return "\n".join(f"MOVE {m.obj} {m.dst}" for m in moves)
    return respond


def goal_oracle(prompt: str) -> str:
    """Keyword matcher for 'put/move <object> in/into/to <zone>' instructions."""
    objects = json.loads(re.search(r"object ids: (\[.*?\])", prompt).group(1).replace("'", '"'))
    zones = json.loads(re.search(r"zone ids: (\[.*?\])", prompt).group(1).replace("'", '"'))
    instruction = re.search(r"Instruction: (.*)", prompt).group(1).lower()
    goal = {}
    for clause in re.split(r"\band\b|,|;", instruction):
        words = set(re.findall(r"[a-z0-9]+", clause))
        obj = max(objects, key=lambda o: len(words & set(o.lower().split("_"))))
        zone = max(zones, key=lambda z: (z.lower().replace("_", " ") in clause.replace("_", " "),
                                         len(words & set(z.lower().split("_")) - {"zone"})))
        if words & set(obj.lower().split("_")):
            goal[obj] = zone
    return json.dumps({"goal": goal})


# ---------------------------------------------------------------- canned generated programs
# What a model plausibly writes for the offline-synthesis prompts. The first policy
# has two typical bugs; the following ones fix what the feedback pointed out.

POLICY_V1 = '''```python
def policy(v):
    for o in v.objects():
        z = v.goal(o)
        if z is not None and v.region(o) != z:
            return (o, z)
    return None
```'''

POLICY_V2 = '''```python
def policy(v):
    # fixed: only move clear objects, and only into a zone that has room
    for o in v.objects():
        z = v.goal(o)
        if z is not None and not v.done(o) and v.is_clear(o) and v.has_room(z):
            return (o, z)
    # otherwise uncover something by parking a clear object in a free slot
    for o in v.objects():
        if not v.done(o) and v.is_clear(o) and v.free_slots() and v.on(o) not in v.free_slots():
            return (o, v.free_slots()[0])
    return None
```'''

POLICY_V3 = '''```python
def policy(v):
    todo = [o for o in v.objects() if not v.done(o)]
    if not todo:
        return None
    clear = [o for o in todo if v.is_clear(o)]

    def home(o):
        """A final place for o that is available now, else None."""
        z = v.goal(o)
        if z is None:
            return None
        tops = [t for t in v.legal_destinations(o)
                if t in v.objects() and v.done(t) and v.region(t) == z]
        if not v.can_support(o) and tops:
            return tops[0]                    # an object that ends a stack goes on a stack
        if v.has_room(z):
            waiting = [x for x in todo if x != o and v.goal(x) == z and v.can_support(x)]
            if v.can_support(o) or not waiting or tops:
                return z
            return None                       # keep the ground pose for an object that can carry
        return tops[0] if tops else None

    # 1. something can go straight to its final place (objects that can carry first)
    for o in sorted(clear, key=lambda o: (not v.can_support(o), o)):
        d = home(o)
        if d is not None:
            return (o, d)

    # 2. otherwise park the clear object that uncovers something useful soonest
    def depth_to_useful(o):
        n, below = 1, v.on(o)
        while below in v.objects():
            if v.done(below):
                return n
            z = v.goal(below)
            if z is not None and v.has_room(z):
                return n
            n, below = n + 1, v.on(below)
        return n if below in v.zones() else 99

    parked = set(v.free_slots())
    movable = [o for o in clear if v.on(o) not in parked]
    for o in sorted(movable, key=lambda o: (depth_to_useful(o), o)):
        if v.region(o) not in v.zones() and v.on(o) not in v.objects():
            continue                          # alone on the table or in a slot: moving it helps nobody
        if v.free_slots():
            return (o, v.free_slots()[0])
        spare = [z for z in v.zones() if v.has_room(z) and z != v.region(o)]
        if spare:
            return (o, spare[0])
    # 3. last resort: any object with a goal into its zone
    for o in clear:
        if v.goal(o) is not None and v.has_room(v.goal(o)):
            return (o, v.goal(o))
    return None
```'''

HEURISTICS = ['''```python
def h(v):
    return sum(1 for o in v.objects() if v.goal(o) is not None and v.region(o) != v.goal(o))
```''', '''```python
def h(v):
    # objects that still have to move at least once
    return sum(1 for o in v.objects() if not v.done(o))
```''', '''```python
def h(v):
    # not-final objects, plus one extra move for each one buried under something
    # that cannot go straight home
    total = 0
    for o in v.objects():
        if v.done(o):
            continue
        total += 1
        if not v.is_clear(o) and v.goal(o) is not None and not v.has_room(v.goal(o)):
            total += 1
    return total
```''']
