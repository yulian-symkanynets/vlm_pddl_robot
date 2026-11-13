"""Different tasks on the same scene, without a GPU.

    python -m approaches.task_demo
    python -m approaches.task_demo --model qwen3-vl-8b     # also understands free phrasing and Slovak
"""
from __future__ import annotations

import argparse

from .task_service import TaskService

SCENE = {
    "scene_version": 1,
    "objects": [{"id": "red_cube_1", "type": "cube", "color": "red"},
                {"id": "blue_cylinder_1", "type": "cylinder", "color": "blue", "on": "red_cube_1"},
                {"id": "green_cube_1", "type": "cube", "color": "green"},
                {"id": "green_prism_1", "type": "prism", "color": "green"}],
    "zones": [{"id": "red_zone", "color": "red", "capacity": 2},
              {"id": "blue_zone", "color": "blue", "capacity": 2},
              {"id": "green_zone", "color": "green", "capacity": 2}],
    "free_slots": ["tmp_1"],
}
INSTRUCTIONS = [
    "Sort the objects by colour.",
    "Stack the green cube, the red cube and the blue cylinder from bottom to top in the blue zone.",
    "Put the red cube on the green cube, but don't put anything on the table.",
    "Rearrange them without using the temporary slots so that the green cube is under the red cube.",
    "Sort everything by colour, but do not touch the green prism and never put anything on the green cube.",
    "Move the blue cylinder to the red zone and make sure nothing is on the red cube.",
    "Put the cube on the cylinder.",
    "Put the red cube on the green prism.",
    "Polož červenú kocku na zelenú kocku.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--quant", default="4bit")
    args = ap.parse_args()
    llm = None
    if args.model:
        from .llm import RepoVLM
        llm = RepoVLM(args.model, args.quant)
    svc = TaskService(llm, "hybrid" if llm else "symbolic", memory_path=None)
    print("scene: blue cylinder on red cube; green cube and green prism on the table; one free slot\n")
    for i, text in enumerate(INSTRUCTIONS):
        print(f"> {text}")
        try:
            r = svc.task(f"demo{i}", {"scene": SCENE, "instruction": text})
        except Exception as e:
            print(f"  rejected: {e}\n")
            continue
        u = r["understanding"]
        if r["status"] == "clarify":
            print(f"  needs clarification ({u['source']}): {r['question']}\n")
            continue
        print(f"  understood by {u['source']} in {u['ms']:.2f} ms, model calls {u['llm_calls']}: {r['task_description']}")
        if r["status"] not in ("ok", "partial"):
            print(f"  {r['status']}: {r['reason']}\n")
            continue
        moves = [f"{s['object']} -> {s['destination']}" for s in r["steps"] if "destination" in s]
        print(f"  plan ({r['source']}, {r['timing_ms']['total_ms']:.2f} ms total): " + "; ".join(moves) + "\n")


if __name__ == "__main__":
    main()
