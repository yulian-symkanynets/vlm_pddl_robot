"""Compare ways of understanding and planning tasks: success, latency, model cost.

    python -m approaches.model_bench                                  # no GPU: rule parser + mock model
    python -m approaches.model_bench --models qwen3-vl-4b qwen3-vl-8b # GPU machine, keys from models.yaml
    python -m approaches.model_bench --models qwen3-vl-8b --modes hybrid vlm_plan --n 200

Every case is a random scene plus an instruction built from a template, with the
correct task known. A run counts as a success only if the returned plan is valid
on the scene AND achieves the correct task (not merely the task the system
understood). Writes approaches/results/model_bench.json.

Configurations:
    rules                 rule parser only, no model
    mock                  scripted model that answers correctly except for injected
                          wrong-id mistakes: tests the correction loop, says nothing about real models
    <model key>           real model; with --modes hybrid (model translates the task, planner plans)
                          and/or vlm_plan (model also writes the plan, validator in the loop)
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

from .llm import MockLLM
from .task_service import TaskService
from semplan.scene import parse_scene, validate
from semplan.task import parse_task
from semplan.task_planner import plan_task

OUT = Path(__file__).parent / "results" / "model_bench.json"
COLOURS = ["red", "blue", "green", "yellow"]
SK = {"red": "červen", "blue": "modr", "green": "zelen", "yellow": "žlt",
      "cube": ("kocku", "ú"), "cylinder": ("valec", "ý"), "prism": ("hranol", "ý")}


def make_scene(rng: random.Random, n: int) -> dict:
    """Objects with unique (colour, shape), so that every phrase names exactly one object."""
    combos = [(c, t) for c in COLOURS for t in ("cube", "cylinder", "prism")]
    rng.shuffle(combos)
    cubes = [x for x in combos if x[1] != "prism"][:max(2, n - 1)]
    chosen = (cubes + [x for x in combos if x not in cubes])[:n]
    rng.shuffle(chosen)
    objs, below = [], None
    for i, (c, t) in enumerate(chosen):
        o = {"id": f"{c}_{t}", "type": t, "color": c}
        if below is not None and rng.random() < 0.45 and below["type"] != "prism":
            o["on"] = below["id"]
        objs.append(o)
        below = o
    return {"scene_version": 1, "objects": objs,
            "zones": [{"id": f"{c}_zone", "color": c, "capacity": 2} for c in COLOURS],
            "free_slots": ["tmp_1", "tmp_2"][:rng.choice([1, 2])]}


def name(o: dict) -> str:
    return f"{o['color']} {o['type']}"


def name_sk(o: dict) -> str:
    noun, ending = SK[o["type"]]
    return f"{SK[o['color']]}{ending} {noun}"


def make_case(rng: random.Random) -> dict | None:
    scene = make_scene(rng, rng.randint(3, 5))
    objs = scene["objects"]
    flat = [o for o in objs if o["type"] != "prism"]
    a, b = rng.sample(flat, 2)
    c = rng.choice([o for o in objs if o not in (a, b)])
    zone = rng.choice(scene["zones"])["id"]
    zname = zone.replace("_", " ")
    templates = [
        ("sort", "Sort the objects by colour.", {"goal": [{"sort_by_color": True}]}),
        ("on", f"Put the {name(a)} on the {name(b)}.", {"goal": [{"object": a["id"], "on": b["id"]}]}),
        ("zone", f"Move the {name(c)} to the {zname}.", {"goal": [{"object": c["id"], "in": zone}]}),
        ("stack_list", f"Stack the {name(a)}, the {name(b)} and the {name(c)} from bottom to top in the {zname}.",
         {"goal": [{"stack": [a["id"], b["id"], c["id"]], "in": zone}]}),
        ("tower_words", f"Build a tower with the {name(a)} at the bottom, the {name(b)} in the middle "
                        f"and the {name(c)} at the very top.",
         {"goal": [{"stack": [a["id"], b["id"], c["id"]]}]}),
        ("on_no_table", f"Put the {name(a)} on the {name(b)}, but don't put anything on the table.",
         {"goal": [{"object": a["id"], "on": b["id"]}], "constraints": {"no_temporary": True}}),
        ("sort_keep", f"Sort everything by colour, but do not touch the {name(c)}.",
         {"goal": [{"sort_by_color": True}], "constraints": {"keep": [c["id"]]}}),
        ("sort_never_on", f"Tidy up by colour and never put anything on the {name(a)}.",
         {"goal": [{"sort_by_color": True}], "constraints": {"never_on": [["*", a["id"]]]}}),
        ("paraphrase", f"I'd like the {name(b)} to end up underneath the {name(a)}; leave the temporary spots empty.",
         {"goal": [{"object": a["id"], "on": b["id"]}], "constraints": {"no_temporary": True}}),
        ("slovak", f"Polož {name_sk(a)} na {name_sk(b)}.", {"goal": [{"object": a["id"], "on": b["id"]}]}),
    ]
    kind, instruction, task = rng.choice(templates)
    try:                                                   # keep only cases with a solvable correct task
        s = parse_scene(dict(scene, task=json.loads(json.dumps(task))))
    except Exception:
        return None
    if plan_task(s, 200).status != "ok" or s.goal_reached():
        return None
    return {"kind": kind, "instruction": instruction, "scene": scene, "task": task}


def same_task(scene: dict, a: dict, b: dict) -> bool:
    s = parse_scene(scene)
    ta, tb = (parse_task(json.loads(json.dumps(t)), s.objects, s.locs) for t in (a, b))
    return (set(ta.atoms), ta.sort, set(ta.never_on), ta.keep, ta.no_temporary, ta.max_height) == \
           (set(tb.atoms), tb.sort, set(tb.never_on), tb.keep, tb.no_temporary, tb.max_height)


def mock_model(cases: list[dict], error_rate: float, seed: int = 0) -> MockLLM:
    """Answers with the correct task; with probability `error_rate` the first answer uses a
    wrong object id (as real models do), which the validation loop then reports back."""
    rng = random.Random(seed)
    by_instruction = {c["instruction"]: c["task"] for c in cases}

    def respond(prompt: str) -> str:
        if "Task JSON:" in prompt:                         # plan request (vlm_plan mode)
            from .mock import planner_oracle
            return planner_oracle(0.0)(prompt)
        instruction = prompt.rsplit("Instruction: ", 1)[1].split("\n", 1)[0]   # the last one is the real one
        answer = json.dumps(by_instruction[instruction])
        if "was rejected" not in prompt and rng.random() < error_rate:
            answer = answer.replace("_cube", "_block", 1)
        return answer
    return MockLLM(respond)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def evaluate(label: str, service: TaskService, cases: list[dict]) -> dict:
    rows, per_kind, errors = [], {}, []
    for i, case in enumerate(cases):
        t0 = time.perf_counter()
        try:
            r = service.task(f"{label}-{i}", {"scene": case["scene"], "instruction": case["instruction"]})
        except Exception as e:                             # a server error counts as a failure
            r = {"status": "error", "reason": repr(e), "task": None, "steps": [], "llm_calls": {}}
            errors.append(repr(e)[:160])
        ms = (time.perf_counter() - t0) * 1e3
        understood = r.get("task") is not None
        match = understood and same_task(case["scene"], r["task"], case["task"])
        success = False
        if r["status"] in ("ok", "partial") and r["steps"]:
            from semplan.scene import Move
            moves = [Move(s["object"], s["destination"]) for s in r["steps"] if "destination" in s]
            truth = parse_scene(dict(case["scene"], task=json.loads(json.dumps(case["task"]))))
            success = validate(truth, moves).ok
        calls = sum((r.get("llm_calls") or {}).values())
        rows.append((understood, match, success, r["status"] == "clarify", calls, ms,
                     (r.get("understanding") or {}).get("source", "none")))
        k = per_kind.setdefault(case["kind"], [0, 0])
        k[0] += success
        k[1] += 1
    n = len(rows)
    out = {"cases": n, "understood": sum(r[0] for r in rows) / n, "task_correct": sum(r[1] for r in rows) / n,
           "plan_success": sum(r[2] for r in rows) / n, "clarify": sum(r[3] for r in rows) / n,
           "llm_calls_per_task": sum(r[4] for r in rows) / n,
           "p50_ms": pct([r[5] for r in rows], .5), "p95_ms": pct([r[5] for r in rows], .95),
           "understood_by": {s: sum(1 for r in rows if r[6] == s) for s in sorted({r[6] for r in rows})},
           "success_by_kind": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per_kind.items())}}
    if errors:
        out["server_errors"] = len(errors)
        out["first_error"] = errors[0]
        print(f"  {len(errors)} request(s) raised an error, first: {errors[0]}")
    llm = service.llm
    if llm is not None and hasattr(llm, "generate_s"):
        out["model_generate_s_per_task"] = llm.generate_s / n
    try:
        import torch
        if torch.cuda.is_available():
            out["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    except ImportError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[], help="model keys from models.yaml")
    ap.add_argument("--modes", nargs="*", default=["hybrid"], choices=["hybrid", "vlm_plan"])
    ap.add_argument("--quant", default="4bit")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    cases = []
    while len(cases) < args.n:
        c = make_case(rng)
        if c:
            cases.append(c)
    res = {"meta": {"system": platform.platform(), "date": time.strftime("%Y-%m-%d"), "cases": len(cases),
                    "seed": args.seed}, "configs": {}}
    configs = [("rules (no model)", lambda: TaskService(None, "symbolic", memory_path=None, use_cache=False)),
               ("mock model, hybrid", lambda: TaskService(mock_model(cases, 0.25), "hybrid", order="llm",
                                                          memory_path=None, use_cache=False)),
               ("rules then mock model", lambda: TaskService(mock_model(cases, 0.25), "hybrid", order="rules,llm",
                                                             memory_path=None, use_cache=False)),
               ("mock model, vlm_plan", lambda: TaskService(mock_model(cases, 0.25), "vlm_plan", order="llm",
                                                            memory_path=None, use_cache=False))]
    for key in args.models:
        from .llm import RepoVLM
        llm = RepoVLM(key, args.quant)
        for mode in args.modes:
            configs.append((f"{key}, {mode}", lambda llm=llm, mode=mode: TaskService(
                llm, mode, order="llm", memory_path=None, use_cache=False)))
            if mode == "hybrid":
                configs.append((f"rules then {key}", lambda llm=llm: TaskService(
                    llm, "hybrid", order="rules,llm", memory_path=None, use_cache=False)))
    for label, make in configs:
        service = make()
        if service.llm is not None and hasattr(service.llm, "generate_s"):
            service.llm.generate_s = 0.0
        res["configs"][label] = r = evaluate(label, service, cases)
        print(f"{label:<28} task correct {r['task_correct']:.0%}  plan success {r['plan_success']:.0%}  "
              f"clarify {r['clarify']:.0%}  calls/task {r['llm_calls_per_task']:.2f}  "
              f"p50 {r['p50_ms']:.2f} ms  p95 {r['p95_ms']:.2f} ms", flush=True)
        print(f"{'':<28} by kind: {r['success_by_kind']}", flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
