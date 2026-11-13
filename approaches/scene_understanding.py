"""Scene understanding: whatever the laptop or the camera produced -> one checked scene message.

The planner needs one small fact per object: what it rests on. The robot side may
describe the table in many ways, so this module reads fields by meaning, not by one
fixed schema:

    planner format     objects with "on", zones with colour and capacity, free_slots
    open-world / VLM   objects with labels, "places", a "support" map, "hidden"
    robot-lab export   shape, position (m, centre), size, height, "relations" on_top_of,
                       "destinations" {id: {x, y, color}}, robot.held_object
    plain detections   {"detections": [{"label": "red cube", "xyz": [...], "score": 0.9}]}
    images             VLM -> open-world JSON -> the same path (perceive())

Steps
  1. objects, places, relations from whichever keys are present
  2. colour and shape from explicit fields, else from the label words
  3. support: explicit relation  >  geometry (position, size)  >  "table"
  4. zone of each stack's bottom object from its position when not given
  5. checks: a fact that is physically impossible is never "repaired" by guessing.
     Every object it involves gets pose_confidence 0, so the planner leaves it alone
     and the response asks the robot side to look again (VERIFY_SCENE).

Every inference is listed in `assumptions`, so the robot side (and an evaluation) can
see what came from the input and what the server concluded.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from semplan.scene import FLAT_TOP, SceneError, parse_scene

COLOURS = ["red", "blue", "green", "yellow", "orange", "purple", "black", "white",
           "pink", "brown", "grey", "cyan", "magenta"]
COLOUR_ALIASES = {"gray": "grey", "violet": "purple"}
SHAPES = {"cube": "cube", "block": "cube", "box": "cube", "cuboid": "cube", "brick": "cube",
          "cylinder": "cylinder", "can": "cylinder", "puck": "cylinder", "disc": "cylinder",
          "prism": "prism", "triangle": "prism", "wedge": "prism", "pyramid": "prism",
          "ball": "ball", "sphere": "ball"}
SLOT_KINDS = {"buffer", "slot", "temporary", "temp", "parking", "free", "free_slot"}
SPECIAL = {"table", "gripper", "unknown"}

DEFAULT_SIZE_M = 0.04          # object footprint when the input gives none
DEFAULT_ZONE_RADIUS_M = 0.07   # half the side of a zone paper when the input gives none
DEFAULT_ZONE_CAPACITY = 3
DEFAULT_SLOT_RADIUS_M = 0.03   # a temporary slot holds one object
MIN_LIFT_M = 0.01              # "above" means at least this much higher
OFF_CENTRE = 0.7               # centre offset / touching distance beyond which a support is doubted


@dataclass
class SceneReading:
    scene: dict                                     # planner scene message
    source_format: str
    assumptions: list[str] = field(default_factory=list)
    uncertain: dict[str, str] = field(default_factory=dict)   # object id -> reason

    def report(self) -> dict:
        return {"source_format": self.source_format, "assumptions": self.assumptions,
                "uncertain": self.uncertain, "objects": len(self.scene["objects"]),
                "zones": len(self.scene["zones"]), "free_slots": len(self.scene["free_slots"])}


# ---------------------------------------------------------------- small readers

def _make_id(text) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")
    return s or "obj"


def _first(d: dict, *keys):
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return None


def _words(*texts) -> list[str]:
    out = []
    for t in texts:
        if isinstance(t, str):
            out += re.findall(r"[a-z]+", t.lower().replace("_", " "))
    return out


def _colour_of(value, *texts) -> str | None:
    if isinstance(value, str) and value.strip():
        v = value.strip().lower()
        v = COLOUR_ALIASES.get(v, v)
        hit = next((COLOUR_ALIASES.get(w, w) for w in _words(v) if COLOUR_ALIASES.get(w, w) in COLOURS), None)
        return hit or v
    for w in _words(*texts):
        w = COLOUR_ALIASES.get(w, w)
        if w in COLOURS:
            return w
    return None


def _shape_of(value, *texts) -> str | None:
    for w in _words(value if isinstance(value, str) else None, *texts):
        if w in SHAPES:
            return SHAPES[w]
        if w.endswith("s") and w[:-1] in SHAPES:
            return SHAPES[w[:-1]]
    return None


def _xyz(o: dict) -> list[float] | None:
    p = _first(o, "position", "xyz", "center", "centre", "pose", "location_xyz")
    if isinstance(p, dict):
        p = [p.get("x"), p.get("y"), p.get("z")]
    if p is None and all(isinstance(o.get(k), (int, float)) for k in ("x", "y")):
        p = [o["x"], o["y"], o.get("z", 0.0)]
    if isinstance(p, (list, tuple)) and len(p) >= 2 and all(isinstance(v, (int, float)) for v in p[:3]):
        return [float(v) for v in p[:3]] + ([0.0] if len(p) == 2 else [])
    return None


def _dims(o: dict) -> tuple[float | None, float | None]:
    """(footprint, height) in input units."""
    size = _first(o, "size", "width", "diameter", "footprint")
    height = _first(o, "height")
    dims = _first(o, "dims", "dimensions", "extent")
    if isinstance(dims, dict):
        dims = [dims.get("x"), dims.get("y"), dims.get("z")]
    if isinstance(dims, (list, tuple)) and len(dims) >= 3:
        size = size if size is not None else max(v for v in dims[:2] if isinstance(v, (int, float)))
        height = height if height is not None else dims[2]
    if isinstance(size, (list, tuple)):
        size = max(size[:2])
    size = float(size) if isinstance(size, (int, float)) else None
    height = float(height) if isinstance(height, (int, float)) else None
    return size, height


# ---------------------------------------------------------------- the reader

def understand_scene(raw: dict) -> SceneReading:
    """Read a scene description in any supported form. Raises SceneError only when the
    input has no usable objects at all; everything else becomes an assumption or an
    'uncertain' mark."""
    if not isinstance(raw, dict):
        raise SceneError("the scene must be a JSON object")
    notes: list[str] = []
    uncertain: dict[str, str] = {}

    # ---- objects
    objs_in = _first(raw, "objects", "detections", "items")
    if objs_in is None and isinstance(raw.get("support"), dict):
        objs_in = [{"id": k} for k in raw["support"]]
    if isinstance(objs_in, dict):
        objs_in = [{"id": k, **(v if isinstance(v, dict) else {})} for k, v in objs_in.items()]
    if not objs_in:
        raise SceneError("the scene has no objects")

    fmt = _guess_format(raw, objs_in)
    scale = _unit_scale(raw, objs_in, notes)

    objects: dict[str, dict] = {}
    alias: dict[str, str] = {}
    geo: dict[str, tuple] = {}                       # id -> (x, y, z, size, height) in metres
    explicit: dict[str, str] = {}                    # id -> support as written (still a name)
    counters: dict[str, int] = {}
    for o in objs_in:
        o = o if isinstance(o, dict) else {"id": o}
        label = _first(o, "label", "name", "class", "category", "desc", "description")
        if any(k in o for k in ("color", "colour")) and _first(o, "color", "colour") is None:
            colour = None                            # the camera said "unknown": do not read it from the id
        else:
            colour = _colour_of(_first(o, "color", "colour"), label, _first(o, "id", "object_id"))
        shape = _shape_of(_first(o, "type", "shape", "class", "category"), label, _first(o, "id"))
        oid = _first(o, "id", "object_id", "uid")
        if oid is None:
            base = "_".join(x for x in (colour, shape) if x) or _make_id(label or "object")
            counters[base] = counters.get(base, 0) + 1
            oid = f"{base}_{counters[base]}"
            notes.append(f"{oid}: no id given, named from its description")
        oid = str(oid)
        if oid in objects:
            k = 2
            while f"{oid}_{k}" in objects:
                k += 1
            notes.append(f"duplicate id {oid}: second one renamed {oid}_{k}")
            oid = f"{oid}_{k}"
        e = {"id": oid, "color": colour}
        if shape is None:
            e["type"] = "unknown"
            e["can_support"] = False
            notes.append(f"{oid}: shape unknown, nothing will be placed on it")
        else:
            e["type"] = shape
        if isinstance(o.get("can_support"), bool):
            e["can_support"] = o["can_support"]
        conf = _first(o, "confidence", "score", "conf", "probability")
        if isinstance(conf, (int, float)):
            e["confidence"] = float(conf)
        for k in ("pose_confidence", "reachable", "blocked_by"):
            if k in o:
                e[k] = o[k]
        if colour is None:
            notes.append(f"{oid}: colour unknown, it will not be sorted until it is identified")
        p = _xyz(o)
        if p is not None:
            size, height = _dims(o)
            p = [v * scale for v in p]
            e["position"] = [round(v, 5) for v in p]
            h_m = height * scale if height else (o["height_mm"] / 1000.0
                                                 if isinstance(o.get("height_mm"), (int, float)) else None)
            geo[oid] = (p[0], p[1], p[2], size * scale if size else None, h_m)
        sup = _first(o, "on", "support", "on_top_of", "rests_on", "parent", "in", "zone", "location")
        if isinstance(sup, str):
            explicit[oid] = sup
        objects[oid] = e
        for a in (oid, label, _first(o, "id"), _first(o, "name")):
            if isinstance(a, str) and a:
                alias.setdefault(a.strip().lower(), oid)
                alias.setdefault(_make_id(a), oid)

    # ---- places: colour zones and free slots
    zones: dict[str, dict] = {}
    zone_geo: dict[str, tuple] = {}                  # zones and slots with a position: (x, y, radius)

    def place_geo(pid, p, default_r):
        c = _xyz(p) if isinstance(p, dict) else None
        if c is None:
            return
        size = _first(p, "size", "width", "diameter")
        r = _first(p, "radius")
        r = (r * scale) if isinstance(r, (int, float)) else \
            (size * scale / 2 if isinstance(size, (int, float)) else default_r)
        zone_geo[pid] = (c[0] * scale, c[1] * scale, r)
    slots: list[str] = []
    places = []
    for key in ("zones", "places", "destinations", "targets"):
        v = raw.get(key)
        if isinstance(v, dict):
            v = [{"id": k, **(d if isinstance(d, dict) else {"color": d})} for k, d in v.items()]
        if isinstance(v, list):
            places += [p if isinstance(p, dict) else {"id": p} for p in v]
    for p in places:
        pid = str(_first(p, "id", "name") or _make_id(_first(p, "label") or "zone"))
        if _make_id(pid) in ("table", "table_1"):
            continue
        label = _first(p, "label", "name")
        kind = _make_id(_first(p, "kind", "type") or "zone")
        colour = _colour_of(_first(p, "color", "colour"), label, pid)
        if kind in SLOT_KINDS or colour in SLOT_KINDS or (colour is None and "temp" in pid.lower()):
            slots.append(pid)
            place_geo(pid, p, DEFAULT_SLOT_RADIUS_M)
            notes.append(f"{pid}: treated as a temporary slot (one object)")
            alias.setdefault(pid.lower(), pid)
            continue
        if colour is None:
            notes.append(f"{pid}: place without a colour, used only as an explicit target")
            colour = pid
        cap = _first(p, "capacity", "slots", "max_objects")
        if not isinstance(cap, (int, float)) or cap < 1:
            cap = DEFAULT_ZONE_CAPACITY
            notes.append(f"{pid}: capacity not given, assumed {cap}")
        zones[pid] = {"id": pid, "color": colour, "capacity": int(cap)}
        place_geo(pid, p, DEFAULT_ZONE_RADIUS_M)
        for a in (pid, label):
            if isinstance(a, str):
                alias.setdefault(a.strip().lower(), pid)
                alias.setdefault(_make_id(a), pid)
    for s in raw.get("free_slots") or raw.get("temporary_slots") or []:
        sid = s if isinstance(s, str) else _first(s, "id", "name")
        if sid and sid not in slots:
            slots.append(str(sid))
            alias.setdefault(str(sid).lower(), str(sid))
            place_geo(str(sid), s, DEFAULT_SLOT_RADIUS_M)
    colours = [z["color"] for z in zones.values()]
    for c in {c for c in colours if colours.count(c) > 1}:
        notes.append(f"several zones are {c}: objects are sorted into the first of them")
        for z in [z for z in zones.values() if z["color"] == c][1:]:
            z["color"] = f"{c}:{z['id']}"

    def name(x):
        if not isinstance(x, str):
            return None
        k = x.strip().lower()
        if k in ("table", "desk", "floor", "ground"):
            return "table"
        if k in ("gripper", "__gripper__", "hand", "tool", "held"):
            return "gripper"
        if k in ("unknown", "none", "null", "?", ""):
            return "unknown"
        return alias.get(k) or alias.get(_make_id(k))

    # ---- explicit support facts (relations and support maps join the per-object fields)
    sup_map = raw.get("support")
    if isinstance(sup_map, list):
        sup_map = {s.get("object"): s.get("on", s.get("in")) for s in sup_map if isinstance(s, dict)}
    for k, v in (sup_map or {}).items():
        x = name(k)
        if x in objects and isinstance(v, str):
            explicit.setdefault(x, v)
    for r in raw.get("relations") or []:
        if isinstance(r, (list, tuple)) and len(r) == 3:
            r = {"subject": r[0], "relation": r[1], "object": r[2]}
        if not isinstance(r, dict):
            continue
        rel = _make_id(_first(r, "type", "relation", "predicate", "rel") or "on")
        a, b = _first(r, "subject", "top", "child", "a"), _first(r, "object", "bottom", "support", "target", "b")
        if rel in ("under", "below", "beneath", "supports", "carries"):
            a, b = b, a
            rel = "on"
        if rel in ("on", "on_top_of", "ontop", "ontopof", "rests_on", "stacked_on",
                   "in", "inside", "in_zone", "at", "within"):
            x = name(a)
            if x in objects and isinstance(b, str):
                if x in explicit and name(explicit[x]) != name(b) and rel.startswith("in"):
                    continue                     # "in zone" is weaker than a direct support
                explicit[x] = b
    held = _first(raw, "holding", "held", "held_object")
    if held is None and isinstance(raw.get("robot"), dict):
        held = _first(raw["robot"], "held_object", "holding", "held")
    if isinstance(held, str) and name(held) in objects:
        explicit[name(held)] = "gripper"

    # ---- geometry: who rests on whom, and which zone a stack stands in
    on: dict[str, str] = {}
    geometric = _support_from_geometry(geo, explicit, objects, zone_geo, notes, uncertain, name)
    for x in objects:
        if x in explicit:
            y = name(explicit[x])
            if y is None:
                uncertain[x] = f"rests on {explicit[x]!r}, which is not in the scene"
                on[x] = "table"
                continue
            if y == "unknown":
                uncertain[x] = "support not known"
                on[x] = "table"
                continue
            g = geometric.get(x)
            if g is not None and y not in ("gripper",) and g != y and not _same_region(g, y, zones):
                # the input says one thing and the positions another
                if y in objects or g in objects:
                    uncertain[x] = f"input says it rests on {y}, positions say {g}"
            on[x] = y
        elif x in geometric:
            on[x] = geometric[x]
        else:
            on[x] = "table"
            if fmt not in ("planner",):
                notes.append(f"{x}: no support and no position given, assumed to stand on the table")

    # ---- physical checks; anything that cannot be true is frozen, not guessed
    _check(on, objects, zones, slots, uncertain, notes)
    for x, why in uncertain.items():
        objects[x]["pose_confidence"] = 0.0
    for x, y in on.items():
        objects[x]["on"] = y

    scene = {k: raw[k] for k in ("scene_id", "task", "goal", "min_confidence", "verified", "occluded",
                                 "free_table") if k in raw}
    if "free_table" not in raw and not slots:
        # Nothing says where objects may be set down. A real table has room, so the robot
        # may put an object down on free table space (it chooses the exact spot).
        scene["free_table"] = True
        notes.append("no free slots given: objects may be put down on free table space (destination 'table')")
    version = _first(raw, "scene_version", "version")
    scene["scene_version"] = int(version) if isinstance(version, (int, float)) else 0
    hidden = [name(h) for h in raw.get("hidden") or [] if name(h) in objects]
    if hidden:
        scene["occluded"] = sorted(set(scene.get("occluded", [])) | set(hidden))
    scene.update(objects=list(objects.values()), zones=list(zones.values()), free_slots=slots)
    parse_scene(scene)                               # must hold after the checks above
    return SceneReading(scene, fmt, notes, uncertain)


def _guess_format(raw: dict, objs: list) -> str:
    first = objs[0] if objs and isinstance(objs[0], dict) else {}
    if "detections" in raw:
        return "detections"
    if "destinations" in raw or raw.get("source") == "simulator_ground_truth":
        return "robot_lab"
    if "support" in raw or "places" in raw:
        return "open_world"
    if "on" in first or ("zones" in raw and "free_slots" in raw):
        return "planner"
    return "generic"


def _unit_scale(raw: dict, objs: list, notes: list) -> float:
    u = str(raw.get("units") or raw.get("unit") or "").lower()
    if u in ("mm", "millimeter", "millimeters", "millimetre", "millimetres"):
        return 0.001
    if u in ("cm", "centimeter", "centimeters"):
        return 0.01
    if u in ("m", "meter", "meters", "metre", "metres"):
        return 1.0
    coords = [abs(v) for o in objs if isinstance(o, dict) and (p := _xyz(o)) for v in p]
    if coords and max(coords) > 3.0:
        notes.append("positions look like millimetres (no 'units' given)")
        return 0.001
    return 1.0


def _support_from_geometry(geo, explicit, objects, zone_geo, notes, uncertain, name) -> dict[str, str]:
    """Support from positions: the highest object below that overlaps in x/y; a stack's
    bottom object belongs to the zone whose area contains it. Works whether z is the
    centre, the bottom or the top of the object, because only the order matters."""
    out: dict[str, str] = {}
    held = {x for x, s in explicit.items() if name(s) == "gripper"}
    pts = {x: g for x, g in geo.items() if x not in held}
    if not pts:
        return out
    inferred = 0
    ground = min(g[2] for g in pts.values())
    for x, (xx, yy, zz, sx, hx) in pts.items():
        below, near = [], []
        for y, (xy_, yy_, zy, sy, hy) in pts.items():
            if y == x:
                continue
            c = 0.5 * ((sx or DEFAULT_SIZE_M) + (sy or DEFAULT_SIZE_M))   # centres this close: edges touch
            lift = max(MIN_LIFT_M, 0.25 * min(hx or DEFAULT_SIZE_M, hy or DEFAULT_SIZE_M))
            dist = math.hypot(xx - xy_, yy - yy_)
            if zz - zy > lift:
                if dist < 0.8 * c:
                    below.append((zy, y, dist / c))
                if dist < 1.3 * c:
                    near.append(y)
        if not below and zz - ground > max(0.6 * (hx or DEFAULT_SIZE_M), 0.025):
            why = f"{x} is {1000 * (zz - ground):.0f} mm above the lowest object but nothing is under it"
            uncertain.setdefault(x, why)
            for y in near:
                uncertain.setdefault(y, why)
        if below:
            below.sort(reverse=True)
            top_z, y, off = below[0]
            if off > OFF_CENTRE:                     # hanging far off its support: ambiguous
                why = f"{x} is far off-centre on {y}; the support may be another object"
                for b in [x] + near:
                    uncertain.setdefault(b, why)
            if len(below) > 1 and abs(below[1][0] - top_z) < MIN_LIFT_M:
                why = f"{x} seems to rest on two objects at once ({y}, {below[1][1]})"
                for b in (x, y, below[1][1]):
                    uncertain.setdefault(b, why)
            if hx is not None and pts[y][4] is not None:
                gap = (zz - top_z) - 0.5 * (hx + pts[y][4])      # centre convention
                if gap > max(0.02, 0.5 * hx) and abs((zz - top_z) - hx) > 0.02 and abs((zz - top_z) - pts[y][4]) > 0.02:
                    notes.append(f"{x}: {gap * 1000:.0f} mm gap above {y}; still treated as resting on it")
            out[x] = y
            inferred += x not in explicit
            continue
        best = None
        for z, (cx, cy, r) in zone_geo.items():
            d = math.hypot(xx - cx, yy - cy)
            if d <= r and (best is None or d < best[0]):
                best = (d, z)
        out[x] = best[1] if best else "table"
        inferred += x not in explicit
    if inferred:
        notes.append(f"supports of {inferred} object(s) derived from positions")
    return out


def _same_region(a: str, b: str, zones: dict) -> bool:
    return a in zones and b in zones and a == b


def _check(on, objects, zones, slots, uncertain, notes) -> None:
    def doubt(x, why):
        uncertain.setdefault(x, why)

    # supports that do not exist as places
    for x, y in list(on.items()):
        if y not in objects and y not in zones and y not in slots and y not in ("table", "gripper"):
            doubt(x, f"rests on unknown {y!r}")
            on[x] = "table"
    if sum(1 for y in on.values() if y == "gripper") > 1:
        for x in [x for x, y in on.items() if y == "gripper"][1:]:
            doubt(x, "reported in the gripper together with another object")
            on[x] = "table"
    # cycles
    for x in list(on):
        seen, cur = {x}, on[x]
        while cur in objects:
            if cur in seen:
                for c in seen:
                    doubt(c, "support cycle")
                    on[c] = "table"
                break
            seen.add(cur)
            cur = on[cur]
    # two objects on one top, or something on a shape without a flat top
    tops: dict[str, list[str]] = {}
    for x, y in on.items():
        if y in objects:
            tops.setdefault(y, []).append(x)
    for y, xs in tops.items():
        flat = objects[y].get("can_support", objects[y].get("type") in FLAT_TOP)
        if len(xs) > 1 or not flat:
            why = f"{' and '.join(sorted(xs))} reported on {y}" if len(xs) > 1 \
                else f"{xs[0]} reported on {y}, which has no flat top"
            doubt(y, why)
            keep = sorted(xs)[0] if flat else None
            for x in xs:
                doubt(x, why)
                if x != keep:
                    on[x] = "table"
            if not flat:
                objects[y]["can_support"] = True     # keep the reported fact, frozen anyway
                on[xs[0]] = y
                notes.append(f"{y}: something rests on it although its shape has no flat top; frozen")
    # zone capacity: what is observed wins over the default
    for z, d in zones.items():
        n = sum(1 for y in on.values() if y == z)
        if n > d["capacity"]:
            notes.append(f"{z}: capacity raised from {d['capacity']} to {n} (that many stacks stand in it)")
            d["capacity"] = n
    for s in slots:
        if sum(1 for y in on.values() if y == s) > 1:
            for x in [x for x, y in on.items() if y == s]:
                doubt(x, f"several objects reported on slot {s}")
    # anything under a doubtful object is doubtful too (it cannot be moved safely)
    for x in list(uncertain):
        cur = on.get(x)
        while cur in objects:
            doubt(cur, f"under {x}, whose position is uncertain")
            cur = on[cur]


# ---------------------------------------------------------------- images

def perceive(image_paths: list, vlm, max_visual_tokens: int = 1024) -> tuple[SceneReading, dict]:
    """Images -> VLM (scene description only, no planning) -> SceneReading.
    `vlm` is anything with generate(paths, prompt, max_visual_tokens=...) -> {"text": ...}."""
    import openworld
    from .llm import extract_json
    out = vlm.generate(list(image_paths), openworld.perceive_prompt(len(image_paths)),
                       max_visual_tokens=max_visual_tokens)
    js = extract_json(out["text"])
    if not isinstance(js, dict):
        raise SceneError("the model did not return a scene description")
    reading = understand_scene(js)
    reading.source_format = "images"
    return reading, {"raw": js, "generate_s": out.get("generate_s"), "text": out["text"]}


# ---------------------------------------------------------------- ids across observations

def match_ids(old: list[dict], new: list[dict]) -> dict[str, str]:
    """old id -> new id for the same physical object, when the robot side renumbers objects.

    Same id: same object. Otherwise an object is matched to one of the unmatched new
    objects with the same shape and colour (an unknown colour matches any colour),
    nearest position first. Without positions only an unambiguous kind is matched.
    Objects that cannot be matched are left out; the caller must not guess them.
    """
    new_ids = {o["id"] for o in new}
    out = {o["id"]: o["id"] for o in old if o["id"] in new_ids}
    free_new = [o for o in new if o["id"] not in out.values()]
    rest = [o for o in old if o["id"] not in out]

    def kind_ok(a, b):
        if a.get("type") != b.get("type"):
            return False
        return a.get("color") is None or b.get("color") is None or a.get("color") == b.get("color")

    pairs = []
    for a in rest:
        cands = [b for b in free_new if kind_ok(a, b)]
        held = [b for b in cands if a.get("on") == "gripper" and b.get("on") == "gripper"]
        if held:                                     # one gripper: held before and now = same object
            pairs.append((-1, 0.0, a["id"], held[0]["id"]))
            continue
        pa = a.get("position")
        for b in cands:
            pb = b.get("position")
            # Order: same known colour before an unknown colour (an unknown colour is
            # only a guess), then positioned pairs by distance, then pairs where one side
            # has no position (typically: in the gripper) take what is left of their kind.
            colour_rank = 0 if a.get("color") is not None and b.get("color") is not None else 2
            if pa and pb:
                pairs.append((colour_rank, math.dist(pa[:2], pb[:2]), a["id"], b["id"]))
            elif pa or pb or (len(cands) == 1 and sum(kind_ok(x, b) for x in rest) == 1):
                pairs.append((colour_rank + 1, 0.0, a["id"], b["id"]))
    used_new: set[str] = set()
    for _, _, a, b in sorted(pairs):
        if a not in out and b not in used_new:
            out[a] = b
            used_new.add(b)
    return out


def rename_ids(value, mapping: dict[str, str]):
    """Replace object ids anywhere inside a task message (goals, stacks, constraints)."""
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [rename_ids(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k: rename_ids(v, mapping) for k, v in value.items()}
    return value


def task_ids(task) -> set[str]:
    """All strings in a task message (a superset of the object ids it names)."""
    if isinstance(task, str):
        return {task}
    if isinstance(task, list):
        return set().union(*(task_ids(v) for v in task)) if task else set()
    if isinstance(task, dict):
        return set().union(*(task_ids(v) for v in task.values())) if task else set()
    return set()
