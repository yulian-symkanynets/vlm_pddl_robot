"""Goal "one stack per colour" (stack_by_color): dedicated search, verification, replanning.
Run: python -m pytest tests/test_colour_stacks.py -q"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import colour_stacks
import openworld
import pddl_task

ROOT = Path(__file__).resolve().parents[1]
SCENE = json.loads((ROOT / "examples" / "requests" / "15_four_towers_sort.json").read_text())["scene"]
SMALL = {"objects": [{"id": "r1", "color": "red", "on": "table"}, {"id": "b1", "color": "blue", "on": "r1"},
                     {"id": "r2", "color": "red", "on": "b1"}, {"id": "b2", "color": "blue", "on": "table"}]}


def final_stacks(scene, plan):
    sup = {o["id"]: o["on"] for o in scene["objects"]}
    for p in plan:
        if p["action"] == "place-on-table":
            sup[p["args"][0]] = "table"
        elif p["action"] == "place-on-item":
            sup[p["args"][0]] = p["args"][1]
    return colour_stacks.stacks_of(sup, sup)


def test_four_towers_become_four_one_colour_stacks_in_15_moves():
    r = openworld.run(scene=SCENE, goal=[{"stack_by_color": True}])
    assert r["status"] == "ok" and r["planner"] == "colour-stacks" and r["verified"]
    assert r["quality"]["moves"] == 15 and r["quality"]["optimal"]
    colour = {o["id"]: o["color"] for o in SCENE["objects"]}
    stacks = final_stacks(SCENE, r["plan"])
    assert len(stacks) == 4 and all(len({colour[x] for x in st}) == 1 for st in stacks)
    assert {p["action"] for p in r["plan"]} <= {"pick-from-table", "pick-from-item", "place-on-table", "place-on-item"}


def test_small_scene_needs_three_moves():
    moves, optimal = colour_stacks.solve({"r1": "table", "b1": "r1", "r2": "b1", "b2": "table"},
                                         {"r1": "red", "b1": "blue", "r2": "red", "b2": "blue"})
    assert optimal and len(moves) == 3


def test_model_answer_and_reported_goal():
    r = openworld.run(instruction="Rearrange so every stack is one colour.", scene=SMALL, vlm=openworld.RuleMock())
    assert r["status"] == "ok" and r["goal"] == [{"stack_by_color": True}]


def test_single_fact_preconditions_get_the_extra_condition():
    blk = "(:action place-on-table\n    :parameters (?x - item)\n    :precondition (holding ?x)\n    :effect (on-table ?x))"
    assert "(and (holding ?x) (can-table ?x))" in pddl_task._add_pre(blk, "(can-table ?x)")
    r = openworld.run(scene={"objects": [{"id": "red_cube", "on": "table"}, {"id": "blue_cube", "on": "red_cube"},
                                         {"id": "green_cube", "on": "table"}]},
                      goal=[{"object": "red_cube", "on": "green_cube"}],
                      constraints={"never_on": [["blue_cube", "table"]]})
    assert r["status"] == "no_plan"        # blue may not go on the table, and there is nowhere else


def test_objects_in_zones_fall_back_to_the_general_planner():
    z = {"objects": [{"id": "r1", "color": "red", "on": "table"}, {"id": "r2", "color": "red", "on": "box"}],
         "places": [{"id": "box", "capacity": 2}]}
    r = openworld.run(scene=z, goal=[{"stack_by_color": True}])
    assert r["status"] == "ok" and r["planner"] != "colour-stacks" and r["quality"]["moves"] == 1


def test_replan_after_first_move_keeps_the_goal():
    r = openworld.run(scene=SCENE, goal=[{"stack_by_color": True}])
    first = r["plan"][1]
    new = copy.deepcopy(SCENE)
    for o in new["objects"]:
        if o["id"] == first["args"][0]:
            o["on"] = "table" if first["action"] == "place-on-table" else first["args"][1]
    r2 = openworld.replan(r, new)
    assert r2["status"] == "ok" and r2["quality"]["moves"] == 14
