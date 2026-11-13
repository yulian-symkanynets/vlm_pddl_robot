"""Random scene generator for tests and benchmarks (seeded, reproducible)."""
from __future__ import annotations

import random

from .scene import Scene, parse_scene

COLORS = ["red", "blue", "green", "yellow", "orange", "purple"]
TYPES = ["cube", "cube", "cylinder", "prism"]       # prisms cannot carry other objects


def random_scene_dict(rng: random.Random, n_objects: int, n_colors: int = 3,
                      n_buffers: int = 2, max_height: int = 3,
                      zone_capacity: tuple[int, int] | None = None,
                      p_in_zone: float = 0.35, prisms: bool = True) -> dict:
    """A consistent random scene in wire format.

    Objects are dealt into random stacks; each stack starts on the open table or
    inside a random (often wrong) zone. Zone capacity defaults to roughly the
    number of objects per colour, so ground poses are scarce but usually sufficient.
    """
    colors = COLORS[:n_colors]
    objs = []
    for i in range(n_objects):
        t = rng.choice(TYPES if prisms else TYPES[:3])
        objs.append({"id": f"o{i}", "type": t, "color": rng.choice(colors)})
    lo, hi = zone_capacity or (max(1, n_objects // n_colors - 1), max(2, n_objects // n_colors + 1))
    zones = [{"id": f"{c}_zone", "color": c, "capacity": rng.randint(lo, hi)} for c in colors]
    used = {z["id"]: 0 for z in zones}
    order = objs[:]
    rng.shuffle(order)
    i = 0
    while i < len(order):
        h = rng.randint(1, max_height)
        col = order[i:i + h]
        i += len(col)
        base = "table"
        if rng.random() < p_in_zone:
            free = [z["id"] for z in zones if used[z["id"]] < z["capacity"]]
            if free:
                base = rng.choice(free)
                used[base] += 1
        below = base
        k = 0
        while k < len(col):
            o = col[k]
            o["on"] = below
            if o["type"] == "prism" and k < len(col) - 1:     # nothing can sit on a prism
                order[i:i] = col[k + 1:]                      # push the rest back
                break
            below = o["id"]
            k += 1
    return {"scene_id": f"rand{rng.randrange(10**6)}", "scene_version": 0, "objects": objs,
            "zones": zones, "free_slots": [f"tmp{i}" for i in range(n_buffers)]}


def random_scene(rng: random.Random, n_objects: int, **kw) -> Scene:
    return parse_scene(random_scene_dict(rng, n_objects, **kw))


def relabel(d: dict, rng: random.Random) -> dict:
    """Same structure, different surface: new ids, permuted colours, shuffled order."""
    colors = sorted({z["color"] for z in d["zones"]})
    perm = dict(zip(colors, rng.sample(COLORS, len(colors))))
    names = {o["id"]: f"x{rng.randrange(10**6)}_{i}" for i, o in enumerate(d["objects"])}
    names.update({z["id"]: f"area_{perm[z['color']]}" for z in d["zones"]})
    names.update({b: f"slot{rng.randrange(10**6)}_{i}" for i, b in enumerate(d["free_slots"])})
    objs = [dict(o, id=names[o["id"]], color=perm.get(o["color"], o["color"]),
                 on=names.get(o["on"], o["on"])) for o in d["objects"]]
    zones = [dict(z, id=names[z["id"]], color=perm[z["color"]]) for z in d["zones"]]
    rng.shuffle(objs)
    rng.shuffle(zones)
    return dict(d, objects=objs, zones=zones, free_slots=[names[b] for b in d["free_slots"]])
