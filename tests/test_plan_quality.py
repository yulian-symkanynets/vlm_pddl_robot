"""Plan quality certificate, the impossibility precheck, and gated streaming.
Run: python -m pytest tests/test_plan_quality.py -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import openworld
import plan_quality

ZONES = [{"id": "red_zone", "kind": "zone", "capacity": 1}, {"id": "blue_zone", "kind": "zone", "capacity": 1},
         {"id": "green_zone", "kind": "zone", "capacity": 1}]


def scene(support):
    return {"objects": [{"id": x, "label": x.replace("_", " ")} for x in support], "places": ZONES,
            "support": support}


def sort_goal(support):
    return [{"object": x, "in": f"{x.split('_')[0]}_zone"} for x in support]


def test_lower_bound_counts_objects_that_must_move():
    sc = openworld.normalize_scene(scene({"red_cube": "table", "blue_cylinder": "red_cube", "green_cube": "green_zone"}))
    goal, _ = openworld.resolve_goal(sort_goal(sc["support"]), sc)
    assert plan_quality.must_move(sc["support"], sc["items"], goal) == {"red_cube", "blue_cylinder"}
    # "put green on red": the cylinder on the red cube must get out of the way, green must move
    goal, _ = openworld.resolve_goal([{"object": "green_cube", "on": "red_cube"}], sc)
    assert plan_quality.must_move(sc["support"], sc["items"], goal) == {"green_cube", "blue_cylinder"}


def test_stacked_scene_gets_a_proven_shortest_plan():
    support = {"red_cube": "table", "blue_cylinder": "red_cube", "green_cube": "table"}
    r = openworld.run(scene=scene(support), goal=sort_goal(support))
    q = r["quality"]
    assert r["status"] == "ok" and q["optimal"] and q["moves"] == q["lower_bound"] == 3
    assert q["moved_more_than_once"] == []


def test_swap_needs_a_second_move_and_the_certificate_says_it_is_unavoidable():
    support = {"red_cube": "blue_zone", "blue_cube": "red_zone"}
    sc = {"objects": [{"id": x, "label": x.replace("_", " ")} for x in support], "places": ZONES[:2],
          "support": support}
    r = openworld.run(scene=sc, goal=sort_goal(support))
    q = r["quality"]
    assert q["moves"] == 3 and q["lower_bound"] == 2 and q["optimal"] and "astar" in q["proof"]
    assert len(q["moved_more_than_once"]) == 1


def test_impossible_goal_is_answered_without_search():
    support = {"red_cube": "table", "red_cylinder": "table"}
    r = openworld.run(scene=scene(support), goal=sort_goal(support))
    assert r["status"] == "no_plan" and r["planner"] == "precheck"
    assert "red_zone holds 1 objects, the goal puts 2 in it" in r["message"]
    assert r["timings"]["plan_s"] == 0.0


def test_kept_plan_after_progress_keeps_its_proof():
    support = {"red_cube": "table", "blue_cylinder": "red_cube", "green_cube": "table"}
    first = openworld.run(scene=scene(support), goal=sort_goal(support))
    moved = dict(support, blue_cylinder="blue_zone")
    r = openworld.replan(first, scene(moved))
    assert r["decision"] == "kept" and r["quality"]["optimal"] and r["quality"]["moves"] == 2


def test_gated_stream_never_lets_a_wasted_move_through():
    from approaches.speculative import experiment
    res = experiment(n=40, seed=3, error_rate=0.6)
    assert res["thinking=0 gate=optimal"]["extra_moves_total"] == 0
    assert res["thinking=0 gate=optimal"]["valid"] == 40
    assert res["thinking=0 gate=legal"]["extra_moves_total"] > 0           # legality alone is not enough
    assert res["thinking=0 gate=none"]["illegal_moves_sent"] > 0
