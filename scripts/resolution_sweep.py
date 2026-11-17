"""How low can the image resolution go before the plan changes? (GPU machine)

    python scripts/resolution_sweep.py --model qwen3-vl-8b
    python scripts/resolution_sweep.py --model qwen3-vl-8b --levels low 512 medium high full auto
    python scripts/resolution_sweep.py --model mock        # checks the script only; the mock ignores images

For every photographed scene in scenes/scenes.yaml, every view and every resolution level,
the server pipeline (image -> scene JSON -> goal -> plan) runs once, and the result is
compared with the ground truth:

    works        the plan, replayed on the TRUE scene, reaches the goal
    shortest     ... and has as many moves as the optimal ground-truth plan
    support_acc  share of objects whose "stands on" the model got right

Object and place ids must match the ground-truth ids for "works" (the perception prompt asks
for <color>_<shape> ids, which is how scenes.yaml names them). A wrong id counts as a failure.
Writes results/resolution_sweep_<model>.json and prints one line per run.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import image_resolution  # noqa: E402
import openworld  # noqa: E402
from evaluate import ground_truth  # noqa: E402
from planning import simulate  # noqa: E402
from scenes import load_config  # noqa: E402


def score(res: dict, scene, gt: dict) -> dict:
    sup = (res.get("scene") or {}).get("support") or {}
    right = sum(1 for x, y in scene.support.items() if sup.get(x) == y)
    out = {"status": res.get("status"), "support_acc": round(right / len(scene.support), 3),
           "works": False, "shortest": False, "moves": None}
    plan = res.get("plan")
    if res.get("status") == "ok" and plan:
        sim = simulate(gt["problem"], [(a["action"], a["args"]) for a in plan])
        picks = sum(1 for a in plan if a["action"].startswith("pick"))
        gt_picks = sum(1 for a, _ in gt["plan"] or [] if a.startswith("pick"))
        out.update(works=sim["ok"], moves=picks, shortest=sim["ok"] and picks == gt_picks,
                   fail=None if sim["ok"] else sim["fail_reason"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="key from models.yaml, or mock")
    ap.add_argument("--quant", default="4bit")
    ap.add_argument("--levels", nargs="+", default=["low", "medium", "high", "full", "auto"])
    ap.add_argument("--crop", type=float, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--max-visual-tokens", type=int, default=1024, help="budget used for 'full'")
    args = ap.parse_args()
    if args.model == "mock":
        vlm = openworld.RuleMock()
    else:
        from vlm import load_model
        vlm = load_model(args.model, args.quant)
    cfg = load_config()
    rows = []
    for sc in cfg.scenes:
        for task in sc.tasks:
            gt = ground_truth(sc, task)
            for view_set in sc.view_sets:
                for level in args.levels:
                    level = int(level) if str(level).isdigit() else level
                    with tempfile.TemporaryDirectory() as d:
                        imgs = []
                        for v in view_set:                       # copies: prepare() writes next to them
                            p = Path(d) / Path(sc.views[v]).name
                            p.write_bytes(Path(sc.views[v]).read_bytes())
                            imgs.append(p)
                        t0 = time.perf_counter()
                        res = image_resolution.run_with_resolution(
                            openworld.run_until_ok, imgs, level, args.crop, args.max_visual_tokens,
                            attempts=2, instruction=task.instruction, vlm=vlm)
                        seconds = time.perf_counter() - t0
                    r = {"scene": sc.id, "task": task.task_id, "views": "+".join(view_set),
                         "level": str(level), "used": res["resolution"]["used"],
                         "image_tokens": sum(t["approx_tokens"] for t in res["resolution"]["tried"]),
                         "image_bytes": res["resolution"]["tried"][-1]["image_bytes"],
                         "seconds": round(seconds, 2),
                         "perceive_s": res.get("timings", {}).get("perceive_s"),
                         **score(res, sc, gt)}
                    rows.append(r)
                    print(json.dumps(r), flush=True)
    summary = {}
    for level in args.levels:
        rs = [r for r in rows if r["level"] == str(level)]
        if rs:
            summary[str(level)] = {
                "runs": len(rs), "works": sum(r["works"] for r in rs), "shortest": sum(r["shortest"] for r in rs),
                "mean_support_acc": round(sum(r["support_acc"] for r in rs) / len(rs), 3),
                "mean_seconds": round(sum(r["seconds"] for r in rs) / len(rs), 2),
                "mean_image_tokens": round(sum(r["image_tokens"] for r in rs) / len(rs))}
    for k, v in summary.items():
        print(f"{k:>8}: works {v['works']}/{v['runs']}, shortest {v['shortest']}, support {v['mean_support_acc']:.0%}, "
              f"{v['mean_seconds']:.1f} s, ~{v['mean_image_tokens']} image tokens")
    out = ROOT / "results" / f"resolution_sweep_{args.model}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "crop": args.crop, "summary": summary, "rows": rows}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
