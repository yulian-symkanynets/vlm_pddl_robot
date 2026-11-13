"""Measurements for the additional strategies. Writes approaches/results/bench_extra.json.

    python -m approaches.bench_extra          # about 2 minutes
"""
from __future__ import annotations

import json
import math
import platform
import random
import time
from pathlib import Path

from . import run
from .anytime import AnytimeStrategy
from .calibration import calibrate, synthetic_log
from .contingent import build_tree, count_leaves, follow
from .cost_aware import CostAwareStrategy, expected_attempts
from .partial_order import deorder, travel_cost
from .symbolic import GraphStrategy
from semplan.cases import ActionStats
from semplan.gen import random_scene
from semplan.planner import plan_astar, quick_infeasible
from semplan.scene import dump_scene, parse_scene, validate

OUT = Path(__file__).parent / "results" / "bench_extra.json"
GRAPH = GraphStrategy()


def solvable(rng, n, count, buffers=(0, 1, 2), min_moves=1):
    out = []
    while len(out) < count:
        s = random_scene(rng, n, n_colors=3, n_buffers=rng.choice(buffers))
        if quick_infeasible(s):
            continue
        o = run(GRAPH, s)
        if o.status == "ok" and len(o.moves) >= min_moves:
            out.append(s)
    return out


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def true_stats() -> ActionStats:
    """Success statistics as the server would have learned them on a robot where
    temporary slots at the table edge and placing on cylinders are unreliable."""
    st = ActionStats()
    table = {("PLACE", None, "buffer"): 0.70, ("PLACE", None, "cylinder"): 0.80,
             ("PICK", "prism", None): 0.75}
    for action in ("PICK", "PLACE"):
        for typ in ("cube", "cylinder", "prism"):
            for other in ("cube", "cylinder", "prism", "zone", "buffer", "table"):
                p = 0.97
                for (a, t, o), v in table.items():
                    if a == action and t in (None, typ) and o in (None, other):
                        p = min(p, v)
                ok = round(p * 400)
                st.counts[(action, typ, other)] = [ok, 400 - ok]
    return st


def main():
    res = {"meta": {"system": platform.platform(), "python": platform.python_version(),
                    "date": time.strftime("%Y-%m-%d")}}

    # ---------------- partial order
    rng = random.Random(31)
    po = {}
    for n in (4, 6, 8, 10):
        flex, widths, saved, steps, ms = [], [], [], [], []
        for s in solvable(rng, n, 300, min_moves=2):
            moves = run(GRAPH, s).moves
            t0 = time.perf_counter()
            plan = deorder(s, moves)
            ms.append((time.perf_counter() - t0) * 1e3)
            flex.append(plan.flexibility())
            widths.append(len(plan.ready(set())))
            steps.append(len(moves))
            pos = {x: (rng.random(), rng.random()) for x in list(s.objects) + list(s.locs)}
            cost = travel_cost(s, plan, pos)
            def total(order):
                idx = [plan.steps.index(m) for m in order]
                return sum(cost(a, b) for a, b in zip([None] + idx[:-1], idx))
            base, best = total(plan.linearize()), total(plan.linearize(cost))
            if base > 0:
                saved.append(1 - best / base)
        po[str(n)] = {"scenes": len(flex), "mean_steps": sum(steps) / len(steps),
                      "mean_flexibility": sum(flex) / len(flex),
                      "mean_ready_at_start": sum(widths) / len(widths),
                      "mean_travel_saved": sum(saved) / len(saved), "deorder_p50_ms": pct(ms, .5)}
        print("partial order", n, po[str(n)], flush=True)
    res["partial_order"] = po

    # ---------------- cost aware
    rng = random.Random(32)
    stats = true_stats()
    ca = {}
    for n in (4, 5, 6):
        strategy = CostAwareStrategy(stats)
        base_e, cost_e, ms, base_len, cost_len, fallback, differ = [], [], [], [], [], 0, 0
        for s in solvable(rng, n, 200, buffers=(1, 2)):
            g = run(GRAPH, s)
            c = run(strategy, s)
            ms.append(c.ms)
            fallback += "budget exhausted" in c.source
            base_e.append(expected_attempts(s, g.moves, stats))
            cost_e.append(expected_attempts(s, c.moves, stats))
            base_len.append(len(g.moves))
            cost_len.append(len(c.moves))
            differ += g.moves != c.moves
        ca[str(n)] = {"scenes": len(ms), "graph_expected_attempts": sum(base_e) / len(base_e),
                      "cost_aware_expected_attempts": sum(cost_e) / len(cost_e),
                      "graph_moves": sum(base_len) / len(base_len),
                      "cost_aware_moves": sum(cost_len) / len(cost_len),
                      "plans_that_differ": differ, "fallback": fallback,
                      "p50_ms": pct(ms, .5), "p95_ms": pct(ms, .95)}
        print("cost aware", n, ca[str(n)], flush=True)
    res["cost_aware"] = ca

    # ---------------- contingent
    rng = random.Random(33)
    ct = {}
    for hidden in (1, 2):
        leaves, ms, paths, invalid, prefix = [], [], 0, 0, []
        for s in solvable(rng, 6, 200, buffers=(1, 2)):
            ids = rng.sample(sorted(s.objects), hidden)
            d = dump_scene(s)
            truth = {}
            for o in d["objects"]:
                if o["id"] in ids:
                    truth[o["id"]] = o["color"]
                    o["color"], o["confidence"] = None, 0.3
            scene = parse_scene(d)
            t0 = time.perf_counter()
            tree = build_tree(scene, {})
            ms.append((time.perf_counter() - t0) * 1e3)
            leaves.append(count_leaves(tree))
            prefix.append(len(tree["moves"]))
            paths += 1
            if not validate(s, follow(tree, truth)).ok:
                invalid += 1
        ct[str(hidden)] = {"scenes": len(ms), "mean_leaves": sum(leaves) / len(leaves),
                           "max_leaves": max(leaves), "mean_moves_before_observing": sum(prefix) / len(prefix),
                           "paths_checked": paths, "invalid_paths": invalid,
                           "p50_ms": pct(ms, .5), "p95_ms": pct(ms, .95)}
        print("contingent", hidden, ct[str(hidden)], flush=True)
    res["contingent"] = ct

    # ---------------- anytime
    rng = random.Random(34)
    at = {}
    for n in (8, 10):
        scenes = solvable(rng, n, 200)
        opt = []
        for s in scenes:
            a = plan_astar(s, 10 ** 9, 5_000)
            opt.append(len(a.moves) if a.status == "ok" else None)
        rows = []
        for budget in (0, 0.5, 2, 10, 50):
            strategy = AnytimeStrategy(budget)
            lens, ms, optimal, known = [], [], 0, 0
            for s, best in zip(scenes, opt):
                o = run(strategy, s)
                ms.append(o.ms)
                lens.append(len(o.moves))
                if best is not None:
                    known += 1
                    optimal += len(o.moves) == best
            rows.append({"budget_ms": budget, "mean_moves": sum(lens) / len(lens),
                         "optimal": optimal, "of": known, "p50_ms": pct(ms, .5), "p95_ms": pct(ms, .95),
                         "max_ms": max(ms)})
            print("anytime", n, rows[-1], flush=True)
        at[str(n)] = rows
    res["anytime"] = at

    # ---------------- CP-SAT
    try:
        t0 = time.perf_counter()
        from .cpsat import CpSatStrategy
        import ortools.sat.python.cp_model  # noqa: F401
        strategy = CpSatStrategy()
        run(strategy, solvable(random.Random(1), 3, 1)[0])            # first call loads the solver
        first_call_s = time.perf_counter() - t0
        rng = random.Random(35)
        cp = {"first_call_s": first_call_s}
        for n in (4, 5, 6, 8):
            ms, same, solved, bools = [], 0, 0, []
            scenes = solvable(rng, n, 40)
            for s in scenes:
                o = run(strategy, s)
                ms.append(o.ms)
                bools.append(strategy.last_stats.get("booleans", 0))
                if o.valid:
                    solved += 1
                    a = plan_astar(s, 10 ** 9, 5_000)
                    same += a.status == "ok" and len(a.moves) == len(o.moves)
            cp[str(n)] = {"scenes": len(scenes), "valid": solved, "same_length_as_astar": same,
                          "p50_ms": pct(ms, .5), "p95_ms": pct(ms, .95), "mean_booleans": sum(bools) / len(bools)}
            print("cp-sat", n, cp[str(n)], flush=True)
        res["cp_sat"] = cp
    except ImportError as e:
        res["cp_sat"] = {"error": str(e)}

    # ---------------- calibration
    test = synthetic_log(50_000, seed=999)
    cal = []
    for n in (200, 1000, 5000):
        ok_runs, thresholds, accepted, test_err = 0, [], [], []
        for seed in range(50):
            r = calibrate(synthetic_log(n, seed=seed), alpha=0.05, delta=0.1)
            if r["threshold"] is None:
                continue
            acc = [c for conf, c in test if conf >= r["threshold"]]
            err = sum(1 for c in acc if not c) / len(acc)
            thresholds.append(r["threshold"])
            accepted.append(len(acc) / len(test))
            test_err.append(err)
            ok_runs += err <= 0.05
        cal.append({"log_size": n, "runs": 50, "threshold_found": len(thresholds),
                    "mean_threshold": sum(thresholds) / len(thresholds) if thresholds else None,
                    "mean_accepted": sum(accepted) / len(accepted) if accepted else None,
                    "mean_test_error": sum(test_err) / len(test_err) if test_err else None,
                    "runs_within_target": ok_runs})
        print("calibration", cal[-1], flush=True)
    fixed = [c for conf, c in test if conf >= 0.8]
    res["calibration"] = {"rows": cal, "alpha": 0.05, "delta": 0.1,
                          "fixed_0.8": {"accepted": len(fixed) / len(test),
                                        "error": sum(1 for c in fixed if not c) / len(fixed)},
                          "example": calibrate(synthetic_log(1000, seed=1))}
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
