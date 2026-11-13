"""Run the scenarios in examples/scenarios.json through the task server and check every plan.

    python -m approaches.run_scenarios                              # no model: rule parser
    python -m approaches.run_scenarios --text-model ollama:llama3.1 # a local text model
    python -m approaches.run_scenarios --model qwen3-vl-8b          # the VLM (GPU machine)
    python -m approaches.run_scenarios --only "no table"            # scenarios whose name contains this

For each scenario: send scene + instruction to TaskService, then check independently of the
server's own validator:
  * the scene was read correctly (what rests on what = 'start')
  * every PICK takes an object with nothing on it, with an empty gripper
  * every PLACE goes onto a clear object with a flat top, a zone with room, or the table
  * the 'expect' conditions: status, final relations, forbidden places, first move,
    objects that must not move, maximum stack height
Without a model, a sentence the rule parser cannot read is replaced by the scenario's
structured 'task' (reported as "structured"), so planning and constraints are still tested.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from approaches.task_service import TaskService  # noqa: E402

FILE = Path(__file__).resolve().parents[1] / "examples" / "scenarios.json"
NO_FLAT_TOP = {"prism", "ball", "sphere"}


def load(path: Path = FILE) -> dict:
    return json.loads(path.read_text())


def replay(scene: dict, start: dict, steps: list[dict], expect: dict) -> tuple[list[str], dict, set]:
    """Independent replay. Returns (problems, final 'on' map, moved objects)."""
    shape = {o["id"]: (o.get("shape") or o.get("type") or "cube") for o in scene["objects"]}
    cap = {z["id"]: int(z.get("capacity", 3)) for z in scene.get("zones", [])}
    on, held, moved, problems = dict(start), None, set(), []
    forbidden = set(expect.get("no_place_on", []))
    for s in steps:
        a, o = s["action"], s.get("object")
        if a == "PICK":
            if held is not None:
                problems.append(f"step {s['step']}: picks {o} while holding {held}")
            if o in on.values():
                top = next(x for x, y in on.items() if y == o)
                problems.append(f"step {s['step']}: picks {o} but {top} is on it")
            held, on[o] = o, "gripper"
            moved.add(o)
        elif a in ("PLACE", "MOVE_TO_TEMPORARY_LOCATION"):
            d = s["destination"]
            if held != o:
                problems.append(f"step {s['step']}: places {o} without holding it")
            if d in forbidden:
                problems.append(f"step {s['step']}: places {o} on {d}, which the task forbids")
            if d in shape:
                if d in on.values():
                    problems.append(f"step {s['step']}: places {o} on {d}, which is covered")
                if shape[d] in NO_FLAT_TOP:
                    problems.append(f"step {s['step']}: places {o} on {d} ({shape[d]}, no flat top)")
            elif d in cap and sum(1 for y in on.values() if y == d) >= cap[d]:
                problems.append(f"step {s['step']}: {d} is full")
            on[o], held = d, None
            if "max_height" in expect:
                h, x = 1, o
                while on.get(x) in on:
                    h, x = h + 1, on[x]
                if h > expect["max_height"]:
                    problems.append(f"step {s['step']}: stack of {h} > {expect['max_height']}")
    if held is not None:
        problems.append(f"ends while holding {held}")
    return problems, on, moved


def base(on: dict, o: str) -> str:
    while on.get(o) in on:
        o = on[o]
    return on.get(o)


def check(sc: dict, data: dict, resp: dict, read: dict | None) -> list[str]:
    scene = data["scenes"][sc["scene"]]
    start = data["starts"][sc["start"]]
    exp = sc["expect"]
    problems = []
    status = {"done": "ok", "partial": "ok"}.get(resp["status"], resp["status"])
    if status != exp["status"]:
        problems.append(f"status {resp['status']} (expected {exp['status']}): "
                        f"{resp.get('question') or resp.get('reason') or ''}")
        return problems
    if read is not None and read != start:
        wrong = {k: (read.get(k), v) for k, v in start.items() if read.get(k) != v}
        problems.append(f"scene read wrongly (read, true): {wrong}")
    if status != "ok":
        return problems
    p, on, moved = replay(scene, start, resp["steps"], exp)
    problems += p
    for a, b in exp.get("on", []):
        if on.get(a) != b:
            problems.append(f"{a} should be on {b}, is on {on.get(a)}")
    for a, b in exp.get("not_on", []):
        if on.get(a) == b:
            problems.append(f"{a} should not be on {b}")
    for a, z in exp.get("in", []):
        if base(on, a) != z:
            problems.append(f"{a} should be in {z}, is in {base(on, a)}")
    for o in exp.get("unmoved", []):
        if o in moved:
            problems.append(f"{o} was moved, but should stay")
    if "first_pick" in exp:
        picks = [s["object"] for s in resp["steps"] if s["action"] == "PICK"]
        if not picks or picks[0] != exp["first_pick"]:
            problems.append(f"first pick is {picks[0] if picks else None}, expected {exp['first_pick']}")
    return problems


def run_one(sc: dict, data: dict, svc_factory) -> dict:
    scene = data["scenes"][sc["scene"]]
    svc = svc_factory()
    resp = svc.task("s", {"scene": scene, "instruction": sc["instruction"]})
    how = (resp.get("understanding") or {}).get("source", "-")
    if resp["status"] == "clarify" and sc["expect"]["status"] != "clarify" and sc.get("task") \
            and svc.text_llm is None:
        svc = svc_factory()                       # rules could not read it: test planning anyway
        resp = svc.task("s", {"scene": scene, "task": sc["task"]})
        how = "structured"
    ses = svc.planning.sessions.get("s")
    read = {o["id"]: o.get("on", "table") for o in ses.raw["objects"]} if ses and ses.raw else None
    problems = check(sc, data, resp, read)
    return {"name": sc["name"], "ok": not problems, "problems": problems, "understood_by": how,
            "status": resp["status"], "task": resp.get("task"),
            "moves": [f"{s['object']}->{s['destination']}" for s in resp.get("steps", []) if s["action"] != "PICK"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", help="VLM key from models.yaml (understands sentences, GPU machine)")
    ap.add_argument("--quant", default="4bit")
    ap.add_argument("--text-model", help="ollama:<model> for sentences")
    ap.add_argument("--only", help="run scenarios whose name contains this text")
    ap.add_argument("--file", default=str(FILE))
    args = ap.parse_args()
    llm = text_llm = None
    if args.model:
        from approaches.llm import RepoVLM
        llm = RepoVLM(args.model, args.quant)
    if args.text_model:
        from approaches.llm import OllamaLLM
        text_llm = OllamaLLM(args.text_model.split(":", 1)[1])
    data = load(Path(args.file))
    factory = lambda: TaskService(llm, text_llm=text_llm, memory_path=None, use_cache=False)  # noqa: E731
    results = [run_one(sc, data, factory) for sc in data["scenarios"]
               if not args.only or args.only.lower() in sc["name"].lower()]
    for r in results:
        print(f"{'OK ' if r['ok'] else 'XX '} {r['name']:<48} [{r['understood_by']}] {r['status']}")
        if r["moves"]:
            print("      " + " | ".join(r["moves"]))
        for p in r["problems"]:
            print("      !! " + p)
    good = sum(r["ok"] for r in results)
    print(f"\n{good}/{len(results)} scenarios correct")
    sys.exit(0 if good == len(results) else 1)


if __name__ == "__main__":
    main()
