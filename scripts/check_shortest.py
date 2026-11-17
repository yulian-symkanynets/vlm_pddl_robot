"""Independent check of the server's "shortest plan" claims.

    python scripts/check_shortest.py            # 40 random scenes
    python scripts/check_shortest.py --n 200 --max-objects 7

For random scenes and goals (sorting, or stacking three objects) the live pipeline
(openworld.run) plans as usual and reports its quality proof. The same planning problem is
then solved by Fast Downward with BLIND A*: no estimate at all, it tries plans in order of
length, so the first plan it finds is the shortest one by definition. The two lengths must
be equal. Blind search gets slow quickly, so keep scenes small (up to about 7 objects).
"""
from __future__ import annotations

import argparse
import random
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import openworld  # noqa: E402
from planning import parse_problem  # noqa: E402
from scenes import make_problem  # noqa: E402
from semplan.gen import random_scene_dict  # noqa: E402


def blind_length(problem, timeout: float):
    from unified_planning.shortcuts import OneshotPlanner, get_environment
    get_environment().credits_stream = None
    with OneshotPlanner(name="fast-downward", params={"fast_downward_search_config": "astar(blind())"}) as p:
        r = p.solve(problem, timeout=timeout)
    return None if r.plan is None else sum(1 for a in r.plan.actions if a.action.name.startswith("pick"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-objects", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeout", type=float, default=60)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    agree, differ, skipped, cases, by_proof = 0, 0, 0, 0, {}
    while cases < args.n:
        d = random_scene_dict(rng, rng.randint(3, args.max_objects), n_colors=3, n_buffers=0, prisms=False,
                              zone_capacity=(1, 3))
        sc_in = {"objects": [{"id": o["id"], "label": f"{o['color']} {o['type']}"} for o in d["objects"]],
                 "places": [{"id": z["id"], "kind": "zone", "capacity": z["capacity"]} for z in d["zones"]],
                 "support": {o["id"]: o.get("on", "table") for o in d["objects"]}}
        if rng.random() < 0.5:
            goal = [{"object": o["id"], "in": f"{o['color']}_zone"} for o in d["objects"]]
        else:
            ids = rng.sample([o["id"] for o in d["objects"]], 3)
            goal = [{"object": ids[0], "on": ids[1]}, {"object": ids[1], "on": ids[2]}]
        r = openworld.run(scene=sc_in, goal=goal)
        if r["status"] != "ok":
            continue
        cases += 1
        sc = openworld.normalize_scene(sc_in)
        g, _ = openworld.resolve_goal(goal, sc)
        best = blind_length(parse_problem(make_problem("x", sc["env"], sc["items"], sc["support"], g, sc["hidden"])),
                            args.timeout)
        q = r["quality"]
        kind = "counting" if "must move" in (q["proof"] or "") else "optimal search" if q["proof"] else "no proof"
        row = by_proof.setdefault(kind, [0, 0])
        if best is None:
            skipped += 1
        elif best == q["moves"]:
            agree += 1
            row[0] += 1
        else:
            differ += 1
            row[1] += 1
            print(f"DIFFERENT: server {q['moves']} moves ({kind}), blind search {best}", flush=True)
    print(f"{cases} scenes: same length as blind search {agree}, different {differ}, blind search timed out {skipped}")
    print("by proof [same, different]:", by_proof)
    sys.exit(1 if differ else 0)


if __name__ == "__main__":
    main()
