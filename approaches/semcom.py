"""Semantic communication experiment: fewer bytes, same plan?

    python -m approaches.semcom                       # representations of generated scenes, no GPU
    python -m approaches.semcom --model qwen3-vl-8b   # adds the image path on the real photographed scenes

The laptop can describe the same scene to the server at different levels:

    image            the camera picture(s); the server's VLM has to recover the scene
    json_full        every object with pose, confidences and readable keys
    json_min         only what planning reads: id, shape, colour, support (+ low confidences)
    packed           the same as json_min in 2-3 bytes per object, ids replaced by a shared codebook
    task_relevant    json_min restricted to the objects the task can depend on

For every level the experiment measures the payload in bytes and whether the server
produces the same result as with complete information:

    identical        exactly the same sequence of moves as the reference plan
    equivalent       not identical, but valid on the true scene, achieves the task, same length
    valid_longer     valid on the true scene and achieves the task, but uses more moves
    failed           no plan, or a plan that is not valid on the true scene

The reference plan is computed from the complete true scene. "Valid on the true scene"
is always checked against the complete scene, not against what was transmitted.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import platform
import random
import struct
import time
from pathlib import Path

from .model_bench import make_case
from semplan.gen import random_scene_dict
from semplan.planner import plan, quick_infeasible
from semplan.scene import Move, parse_scene, validate
from semplan.task_planner import plan_task

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent / "results" / "semcom.json"
TYPES = ["cube", "cylinder", "prism", "other"]
COLOURS = [None, "red", "blue", "green", "yellow", "orange", "purple", "other"]


def compact(d) -> bytes:
    return json.dumps(d, separators=(",", ":")).encode()


def plan_for(msg: dict):
    """The server's planning step for a scene message. -> (scene, moves or None)"""
    s = parse_scene(json.loads(json.dumps(msg)))
    r = plan_task(s, 300) if s.general else plan(s, astar_ms=100)
    return s, (r.moves if r.status in ("ok", "partial") else None)


# ================================================================ representations

def json_full(msg: dict) -> bytes:
    return compact(msg)


def to_min(msg: dict) -> dict:
    """Keep only the fields the planner reads. Confidences are kept only where they matter."""
    objs = []
    for o in msg["objects"]:
        e = {"id": o["id"], "type": o.get("type", "cube"), "color": o.get("color")}
        if o.get("on", "table") != "table":
            e["on"] = o["on"]
        for k in ("confidence", "pose_confidence"):
            if o.get(k, 1.0) < 0.995:
                e[k] = round(o[k], 2)
        for k in ("reachable", "can_support", "blocked_by"):
            if k in o:
                e[k] = o[k]
        objs.append(e)
    out = {"scene_version": msg.get("scene_version", 0), "objects": objs,
           "zones": msg["zones"], "free_slots": msg.get("free_slots", [])}
    if "task" in msg:
        out["task"] = msg["task"]
    return out


def json_min(msg: dict) -> bytes:
    return compact(to_min(msg))


def _codebook(msg: dict) -> dict[str, str]:
    """Short names that keep the alphabetical order of the original ids (so that the
    planner's tie-breaking, and therefore the plan, does not change)."""
    names = sorted({o["id"] for o in msg["objects"]} | {z["id"] for z in msg["zones"]}
                   | set(msg.get("free_slots", [])))
    return {n: ("n" if n < "table" else "u") + f"{i:02d}" for i, n in enumerate(names)}


def packed(msg: dict) -> tuple[bytes, dict[str, str]]:
    """Binary form. The laptop keeps the codebook (short name -> id); the server works with
    the short names and the laptop translates the returned plan back.

    header: version u16, #objects u8, #zones u8, #slots u8
    object: 1 byte  type(2 bits) colour(3) can_support(1) reachable(1) spare(1)
            1 byte  support (index into objects, zones, slots; 255 = table)
            1 byte  identity confidence, 1 byte pose confidence (0..255)
    zone:   1 byte  colour(3 bits) capacity(5 bits)
    The task travels as compact JSON over the short names.
    """
    cb = _codebook(msg)
    objs = sorted(msg["objects"], key=lambda o: o["id"])
    zones, slots = msg["zones"], msg.get("free_slots", [])
    index = {o["id"]: i for i, o in enumerate(objs)}
    index.update({z["id"]: len(objs) + i for i, z in enumerate(zones)})
    index.update({s: len(objs) + len(zones) + i for i, s in enumerate(slots)})
    out = bytearray(struct.pack("<HBBB", msg.get("scene_version", 0) & 0xFFFF, len(objs), len(zones), len(slots)))
    for o in objs:
        t = TYPES.index(o.get("type", "cube")) if o.get("type", "cube") in TYPES else 3
        c = COLOURS.index(o.get("color")) if o.get("color") in COLOURS else 7
        sup = o.get("can_support", o.get("type", "cube") != "prism")
        out.append(t << 6 | c << 3 | int(bool(sup)) << 2 | int(bool(o.get("reachable", True))) << 1)
        out.append(index.get(o.get("on", "table"), 255))
        out.append(round(255 * o.get("confidence", 1.0)))
        out.append(round(255 * o.get("pose_confidence", 1.0)))
    for z in zones:
        c = COLOURS.index(z["color"]) if z["color"] in COLOURS else 7
        out.append(c << 5 | min(int(z.get("capacity", 1)), 31))
    if "task" in msg:
        out += json.dumps(_rename(msg["task"], cb), separators=(",", ":")).encode()
    return bytes(out), {v: k for k, v in cb.items()}


def _rename(x, cb):
    if isinstance(x, dict):
        return {k: _rename(v, cb) for k, v in x.items()}
    if isinstance(x, list):
        return [_rename(v, cb) for v in x]
    return cb.get(x, x) if isinstance(x, str) else x


def unpack(data: bytes, original: dict) -> dict:
    """Server side of `packed`: rebuild a scene message over the short names. `original` is only
    used for the list of names in codebook order, which both sides share."""
    cb = _codebook(original)
    version, n, nz, ns = struct.unpack_from("<HBBB", data)
    obj_ids = [cb[o["id"]] for o in sorted(original["objects"], key=lambda o: o["id"])]
    zone_ids = [cb[z["id"]] for z in original["zones"]]
    slot_ids = [cb[s] for s in original.get("free_slots", [])]
    names = obj_ids + zone_ids + slot_ids
    objs, p = [], 5
    for i in range(n):
        b, sup, conf, pose = data[p:p + 4]
        p += 4
        e = {"id": obj_ids[i], "type": TYPES[b >> 6], "color": COLOURS[(b >> 3) & 7],
             "can_support": bool(b >> 2 & 1), "reachable": bool(b >> 1 & 1),
             "confidence": conf / 255, "pose_confidence": pose / 255}
        if sup != 255:
            e["on"] = names[sup]
        objs.append(e)
    zones = []
    for i in range(nz):
        b = data[p]
        p += 1
        zones.append({"id": zone_ids[i], "color": COLOURS[b >> 5], "capacity": b & 31})
    msg = {"scene_version": version, "objects": objs, "zones": zones, "free_slots": slot_ids}
    if p < len(data):
        msg["task"] = json.loads(data[p:].decode())
    return msg


def task_relevant(msg: dict) -> dict:
    """Drop the objects the task cannot depend on.

    Kept: every object the task names, every object that still has to be sorted, and the
    complete stack of each of them (what is above must be moved, what is below is uncovered).
    A dropped stack that stands in a zone is replaced by one less ground pose in that zone;
    a dropped object on a free slot removes that slot. The server therefore sees fewer
    options than really exist, never more, so its plan is executable in the real scene.
    """
    s = parse_scene(json.loads(json.dumps(msg)))
    seeds: set[str] = set()
    t = s.task
    if t is not None:
        for a in t.atoms:
            seeds |= {x for x in a[1:] if x in s.objects}
        seeds |= {x for pair in t.never_on for x in pair if x in s.objects}
    goals = s.goals()
    seeds |= set(s.objects) - s.settled(goals) if (t is None or t.sort) else set()
    seeds |= {o for o in s.objects if not s.trusted(o) or not s.movable(o)}
    keep: set[str] = set()
    for base, col in s.stacks():
        if seeds & set(col):
            keep |= set(col)
    # A finished stack in a zone that still expects arrivals is a place to stack on. It is
    # summarised by its top object standing on the zone floor (the buried objects are dropped).
    expecting = {z for o, z in goals.items() if o in keep and s.base(o) != z}
    summary: dict[str, str] = {}
    dropped_floor: dict[str, int] = {}
    for base, col in s.stacks():
        if set(col) & keep or base not in s.locs:
            continue
        if base in expecting and s.objects[col[-1]].can_support and (t is None or t.max_height is None):
            summary[col[-1]] = base
        else:
            dropped_floor[base] = dropped_floor.get(base, 0) + 1
    m = to_min(msg)
    kept_objs = []
    for o in m["objects"]:
        if o["id"] in keep:
            kept_objs.append(o)
        elif o["id"] in summary:
            kept_objs.append(dict(o, on=summary[o["id"]]))
    m["objects"] = kept_objs
    keep = keep | set(summary)
    m["zones"] = [dict(z, capacity=z.get("capacity", 1) - dropped_floor.get(z["id"], 0)) for z in m["zones"]]
    m["free_slots"] = [x for x in m["free_slots"] if dropped_floor.get(x, 0) == 0]
    if "task" in m and "constraints" in m["task"] and "keep" in m["task"]["constraints"]:
        task = json.loads(json.dumps(m["task"]))
        task["constraints"]["keep"] = [x for x in task["constraints"]["keep"] if x in keep]
        m["task"] = task
    return m


# ================================================================ comparison

def classify(truth_msg: dict, ref: list[Move] | None, moves: list[Move] | None) -> str:
    if ref is None:
        return "identical" if moves is None else "failed"
    if moves is None:
        return "failed"
    if moves == ref:
        return "identical"
    truth = parse_scene(json.loads(json.dumps(truth_msg)))
    if not validate(truth, moves).ok:
        return "failed"
    return "equivalent" if len(moves) == len(ref) else "valid_longer"


def enrich(msg: dict, rng: random.Random) -> dict:
    """Give a generated scene the extra fields a real laptop would send."""
    msg = json.loads(json.dumps(msg))
    msg.setdefault("scene_id", f"scene_{rng.randrange(10 ** 4):04d}")
    for o in msg["objects"]:
        o["position"] = [round(rng.uniform(0.15, 0.55), 3), round(rng.uniform(-0.3, 0.3), 3),
                         round(rng.uniform(0.02, 0.15), 3)]
        o["orientation_deg"] = round(rng.uniform(0, 90), 1)
        o["size_mm"] = [40, 40, rng.choice([30, 40, 60])]
        o["confidence"] = round(rng.uniform(0.86, 0.995), 3)
        o["pose_confidence"] = round(rng.uniform(0.9, 0.999), 3)
        o["bbox"] = [rng.randrange(0, 600), rng.randrange(0, 400), rng.randrange(40, 90), rng.randrange(40, 90)]
    for z in msg["zones"]:
        z["position"] = [round(rng.uniform(0.15, 0.55), 3), round(rng.uniform(-0.3, 0.3), 3)]
    return msg


def cases(n: int, seed: int) -> list[dict]:
    """Half instruction-style tasks on small scenes, half colour sorting on 4-10 objects."""
    rng = random.Random(seed)
    out = []
    while len(out) < n // 2:
        c = make_case(rng)
        if c:
            out.append({"kind": "task:" + c["kind"], "msg": enrich(dict(c["scene"], task=c["task"]), rng)})
    while len(out) < n:
        d = random_scene_dict(rng, rng.randint(4, 10), n_colors=3, n_buffers=rng.choice([1, 2]))
        s = parse_scene(d)
        if quick_infeasible(s) or plan(s).status != "ok" or s.goal_reached():
            continue
        out.append({"kind": "sort", "msg": enrich(d, rng)})
    return out


def image_sizes() -> dict:
    """Bytes of the photographs used in the earlier VLM experiments, and of smaller versions."""
    files = sorted((ROOT / "images").glob("*.jpg"))
    if not files:
        return {}
    out = {"original": {"files": len(files), "mean_bytes": sum(f.stat().st_size for f in files) / len(files)}}
    try:
        from PIL import Image
    except ImportError:
        return out
    for width, quality in ((1024, 85), (768, 75), (512, 70), (384, 60), (256, 50)):
        sizes = []
        for f in files:
            with Image.open(f) as im:
                im = im.convert("RGB")
                im = im.resize((width, max(1, round(im.height * width / im.width))))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=quality)
                sizes.append(buf.tell())
        out[f"{width}px q{quality}"] = {"files": len(files), "mean_bytes": sum(sizes) / len(sizes)}
    return out


def synthetic(n: int, seed: int) -> dict:
    levels = ("json_full", "json_min", "packed", "task_relevant")
    stats = {l: {"bytes": [], "gzip": [], "identical": 0, "equivalent": 0, "valid_longer": 0, "failed": 0,
                 "objects_sent": [], "ms": [], "second_round": 0} for l in levels}
    data = cases(n, seed)
    for c in data:
        msg = c["msg"]
        _, ref = plan_for(msg)
        for level in levels:
            if level == "json_full":
                payload, received, back = json_full(msg), msg, None
            elif level == "json_min":
                payload = json_min(msg)
                received, back = json.loads(payload), None
            elif level == "packed":
                payload, back = packed(msg)
                received = unpack(payload, msg)
            else:
                received = task_relevant(msg)
                payload, back = compact(received), None
            t0 = time.perf_counter()
            try:
                _, moves = plan_for(received)
            except Exception:
                moves = None
            st = stats[level]
            if level == "task_relevant" and moves is None and ref is not None:
                # Two-round protocol: the server cannot plan with the subset, asks for the rest.
                st["second_round"] += 1
                received = json.loads(json_min(msg))
                payload = payload + json_min(msg)
                _, moves = plan_for(received)
            st["ms"].append((time.perf_counter() - t0) * 1e3)
            if moves is not None and back is not None:                 # laptop translates short names back
                moves = [Move(back.get(m.obj, m.obj), back.get(m.dst, m.dst)) for m in moves]
            st[classify(msg, ref, moves)] += 1
            st["bytes"].append(len(payload))
            st["gzip"].append(len(gzip.compress(payload)))
            st["objects_sent"].append(len(received["objects"]) / len(msg["objects"]))
    out = {}
    for level, st in stats.items():
        k = len(st["bytes"])
        out[level] = {"cases": k, "mean_bytes": sum(st["bytes"]) / k, "mean_gzip_bytes": sum(st["gzip"]) / k,
                      "identical": st["identical"], "equivalent": st["equivalent"],
                      "valid_longer": st["valid_longer"], "failed": st["failed"],
                      "objects_sent_fraction": sum(st["objects_sent"]) / k,
                      "second_round": st["second_round"],
                      "plan_p50_ms": sorted(st["ms"])[k // 2]}
    return out


# ================================================================ image path on the real scenes (GPU)

def real_scene_messages() -> list[dict]:
    """Ground truth of the photographed scenes (scenes/scenes.yaml) as scene messages."""
    import yaml
    cfg = yaml.safe_load((ROOT / "scenes" / "scenes.yaml").read_text())
    shape = {"block": "cube", "cube": "cube", "cylinder": "cylinder", "prism": "prism"}
    zones = [{"id": p, "color": p.split("_")[0], "capacity": int(d.get("slots", 2))}
             for p, d in cfg["env"]["places"].items()]
    out = []
    for sc in cfg["scenes"]:
        objs = []
        for i in sc["items"]:
            d = cfg["item_defs"][i]
            e = {"id": i, "type": shape.get(d["shape"], "cube"), "color": d["color"]}
            if sc["support"][i] != "table":
                e["on"] = sc["support"][i]
            if i in sc.get("hidden", []):
                e["pose_confidence"] = 0.0
            objs.append(e)
        out.append({"scene_id": sc["id"], "views": sc.get("views", {}),
                    "msg": {"scene_version": 1, "objects": objs, "zones": zones, "free_slots": ["tmp_1", "tmp_2"]}})
    return out


def image_path(model_key: str, quant: str) -> dict:
    """image -> VLM perception -> scene message -> plan, compared with the plan from ground truth.
    Not executed on the development machine (needs the GPU)."""
    import openworld
    from PIL import Image

    from .llm import RepoVLM, extract_json
    from .task_service import from_openworld
    vlm = RepoVLM(model_key, quant)
    rows = []
    import tempfile
    for sc in real_scene_messages():
        _, ref = plan_for(sc["msg"])
        for view, fname in sc["views"].items():
            src = ROOT / "images" / fname
            if not src.exists():
                continue
            for width, quality in ((None, None), (1024, 85), (512, 70), (256, 50)):
                with tempfile.TemporaryDirectory() as d:
                    p = Path(d) / "view.jpg"
                    with Image.open(src) as im:
                        im = im.convert("RGB")
                        if width:
                            im = im.resize((width, max(1, round(im.height * width / im.width))))
                        im.save(p, "JPEG", quality=quality or 95)
                    size = p.stat().st_size
                    t0 = time.perf_counter()
                    out = vlm.model.generate([p], openworld.perceive_prompt(1))
                    seconds = time.perf_counter() - t0
                try:
                    received = from_openworld(extract_json(out["text"]))
                    received["free_slots"] = sc["msg"]["free_slots"]
                    _, moves = plan_for(received)
                    result = classify(sc["msg"], ref, moves)
                except Exception as e:
                    result = f"failed ({type(e).__name__})"
                rows.append({"scene": sc["scene_id"], "view": view, "variant": f"{width or 'original'}px",
                             "bytes": size, "result": result, "model_s": seconds})
                print(rows[-1], flush=True)
    return {"model": model_key, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--model", help="model key from models.yaml: also run the image path (GPU)")
    ap.add_argument("--quant", default="4bit")
    args = ap.parse_args()
    res = {"meta": {"system": platform.platform(), "date": time.strftime("%Y-%m-%d"), "cases": args.n},
           "levels": synthetic(args.n, args.seed), "images": image_sizes()}
    print(f"{'level':<16}{'bytes':>9}{'gzip':>8}{'identical':>11}{'equivalent':>12}{'longer':>8}{'failed':>8}{'objects sent':>14}{'2nd round':>11}")
    for level, r in res["levels"].items():
        print(f"{level:<16}{r['mean_bytes']:>9.0f}{r['mean_gzip_bytes']:>8.0f}{r['identical']:>11}{r['equivalent']:>12}"
              f"{r['valid_longer']:>8}{r['failed']:>8}{r['objects_sent_fraction']:>13.0%}{r['second_round']:>11}")
    for name, r in res["images"].items():
        print(f"image {name:<12}{r['mean_bytes']:>9.0f}   (plan equivalence needs --model)")
    try:
        real = real_scene_messages()
        res["real_scenes"] = [{"scene": r["scene_id"], "json_min_bytes": len(json_min(r["msg"])),
                               "packed_bytes": len(packed(r["msg"])[0]),
                               "plan_moves": len(plan_for(r["msg"])[1] or [])} for r in real]
        print("real scenes (ground truth):", res["real_scenes"])
    except Exception as e:
        print("real scenes skipped:", repr(e))
    if args.model:
        res["image_path"] = image_path(args.model, args.quant)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
