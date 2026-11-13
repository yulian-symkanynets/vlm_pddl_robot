"""Compare all strategies on the same random scenes. Writes approaches/results/bench.json.

    python -m approaches.bench            # about 2 minutes
    python -m approaches.bench --quick

Same scenes, same validator, same clock for every strategy. Strategies marked
"mock" use scripted model answers: their validity only reflects the injected
error rate and their time excludes model inference, so those two columns must
not be read as results about a real model.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

from . import run
from .adaptive import AdaptivePlanner
from .behavior_tree import run_tree
from .htn import HtnStrategy
from .learned_policy import LearnedPolicyStrategy, make_dataset, train
from .llm import MockLLM
from .llm_offline import (HeuristicSearchStrategy, ProgramStrategy, synthesize_heuristic,
                          synthesize_policy)
from .llm_online import LLMDirectStrategy, LLMModuloStrategy
from .mock import HEURISTICS, POLICY_V1, POLICY_V2, POLICY_V3, planner_oracle
from .replanning import compare, receding_horizon_episode
from .symbolic import CaseBasedStrategy, GraphStrategy
from semplan.gen import random_scene
from semplan.planner import plan_astar, quick_infeasible
from semplan.sim import SimRobot

OUT = Path(__file__).parent / "results" / "bench.json"


def solvable(rng, n, count):
    out = []
    while len(out) < count:
        s = random_scene(rng, n, n_colors=3, n_buffers=rng.choice([0, 1, 2]))
        if not quick_infeasible(s) and run(GraphStrategy(), s).status == "ok":
            out.append(s)
    return out


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    q = ap.parse_args().quick
    rng = random.Random(2026)
    N = 60 if q else 300
    res = {"meta": {"system": platform.platform(), "python": platform.python_version(),
                    "date": time.strftime("%Y-%m-%d"), "scenes_per_size": N}}

    # ---- offline preparation (not part of request latency)
    train_scenes = [s for n in (3, 4, 5) for s in solvable(rng, n, 14)]
    t0 = time.perf_counter()
    code, plog = synthesize_policy(MockLLM([POLICY_V1, POLICY_V2, POLICY_V3]), train_scenes, accept=0.8)
    hcode, hlog = synthesize_heuristic(MockLLM(HEURISTICS), train_scenes[:20])
    t1 = time.perf_counter()
    data = make_dataset(100 if q else 400, seed=5)
    weights, acc = train(data)
    t2 = time.perf_counter()
    res["offline"] = {"policy_rounds": plog, "heuristic_candidates": hlog,
                      "learned": {"train_scenes": len(data), "train_states": sum(len(d) for d in data),
                                  "top1_accuracy": acc, "data_and_training_s": t2 - t1}}
    print("offline:", plog[-1], "| learned top-1 %.3f" % acc, flush=True)

    def strategies():
        out = [("graph", GraphStrategy(), "symbolic", N),
               ("astar (optimal)", GraphStrategy(optimal=True, astar_ms=5000), "symbolic", N),
               ("htn", HtnStrategy(), "symbolic", N),
               ("case_based", CaseBasedStrategy(), "symbolic + memory", N),
               ("llm_program", ProgramStrategy(code), "generated code (mock model)", N),
               ("llm_heuristic", HeuristicSearchStrategy(hcode), "generated code (mock model)", N),
               ("learned", LearnedPolicyStrategy(weights, None), "trained ranker", N),
               ("learned+symbolic", LearnedPolicyStrategy(weights), "trained ranker + fallback", N),
               ("adaptive", AdaptivePlanner(), "rule-based router", N),
               ("llm_direct", LLMDirectStrategy(MockLLM(planner_oracle(0.3, 1))), "mock model", N),
               ("llm_modulo", LLMModuloStrategy(MockLLM(planner_oracle(0.3, 1))), "mock model", N)]
        try:
            from .symbolic import PddlStrategy
            out[3:3] = [("pddl:pyperplan", PddlStrategy("pyperplan"), "symbolic", 10 if q else 30),
                        ("pddl:fast-downward", PddlStrategy("fast-downward"), "symbolic", 8 if q else 25)]
        except Exception as e:
            print("PDDL skipped:", e)
        return out

    res["strategies"] = {}
    for n in (4, 6, 8):
        scenes = solvable(rng, n, N)
        opt = {}
        if n <= 6:
            for i, s in enumerate(scenes):
                a = plan_astar(s, 300_000, 5_000)
                if a.status == "ok":
                    opt[i] = len(a.moves)
        for label, stg, kind, count in strategies():
            if label == "pddl:pyperplan" and n > 6:
                count = min(count, 12)
            ms, valid, moves, ratio, calls = [], 0, [], [], 0
            for i, s in enumerate(scenes[:count]):
                o = run(stg, s)
                ms.append(o.ms)
                calls += o.llm_calls
                if o.valid:
                    valid += 1
                    moves.append(len(o.moves))
                    if i in opt and opt[i]:
                        ratio.append(len(o.moves) / opt[i])
            row = {"kind": kind, "scenes": count, "valid": valid, "p50_ms": pct(ms, .5),
                   "p95_ms": pct(ms, .95), "max_ms": max(ms),
                   "mean_moves": sum(moves) / len(moves) if moves else None,
                   "moves_vs_optimal": sum(ratio) / len(ratio) if ratio else None,
                   "llm_calls_per_request": calls / count}
            if label == "learned+symbolic":
                row["fallback_used"] = stg.used_fallback
            if label == "case_based":
                row["case_stats"] = dict(stg.cases.stats)
            res["strategies"].setdefault(label, {})[str(n)] = row
            print(f"n={n} {label:<20} valid {valid}/{count} p50 {row['p50_ms']:.3f} ms "
                  f"p95 {row['p95_ms']:.3f} ms", flush=True)

    # ---- replanning
    res["replanning"] = compare(80 if q else 400, seed=3)
    for r in res["replanning"]:
        print("replan", r["method"], "p50 %.3f ms" % r["p50_ms"], "moves %.2f" % r["mean_moves"], flush=True)
    loop = {"behavior_tree": [0, 0, 0], "receding_horizon": [0, 0, 0]}
    episodes = 0
    lrng = random.Random(11)
    while episodes < (40 if q else 200):
        s = random_scene(lrng, lrng.randint(3, 7), n_colors=3, n_buffers=2)
        if quick_infeasible(s) or run(GraphStrategy(), s).status != "ok":
            continue
        episodes += 1
        robot = SimRobot(s, lrng, p_pick_fail=0.1)
        out = run_tree(robot, disturb=lambda t, r=robot: r.human_disturbance()
                       if lrng.random() < 0.1 and r.truth.held() is None else None)
        loop["behavior_tree"][0] += out["goal"]
        loop["behavior_tree"][1] += out["replans"]
        loop["behavior_tree"][2] += out["steps"]
        out = receding_horizon_episode(SimRobot(s, lrng, p_pick_fail=0.1))
        loop["receding_horizon"][0] += out["goal"]
        loop["receding_horizon"][1] += out["plans"]
        loop["receding_horizon"][2] += out["robot_steps"]
    res["closed_loop"] = {k: {"episodes": episodes, "goal": v[0], "planner_calls_per_episode": v[1] / episodes,
                              "robot_steps_per_episode": v[2] / episodes} for k, v in loop.items()}
    print("closed loop", res["closed_loop"], flush=True)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
