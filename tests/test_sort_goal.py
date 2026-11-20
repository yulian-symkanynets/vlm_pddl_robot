"""Sort-by-colour goals, colours of objects and places, empty model answers (stand-in model).
Run: python -m pytest tests/test_sort_goal.py -q"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import openworld

ROOT = Path(__file__).resolve().parents[1]
REQ = json.loads((ROOT / "examples" / "requests" / "15_four_towers_sort.json").read_text())
SC = openworld.normalize_scene(REQ["scene"])


class FixedAnswer:
    """A stand-in model that always gives the same goal answer, counting its calls."""
    key = "fixed"

    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    def generate(self, image_paths, prompt, *a, **kw):
        self.calls += 1
        return {"text": json.dumps(self.answer), "generate_s": 0.0}


def test_colours_are_kept_and_shown_to_the_model():
    assert SC["items"]["red_cube_1"]["color"] == "red" and SC["env"]["places"]["red_zone"]["color"] == "red"
    prompt = openworld.goal_prompt(SC, REQ["instruction"])
    assert "- red_zone: zone, colour red (room for 4)" in prompt and '{"sort_by_color": true}' in prompt
    plain = openworld.normalize_scene({"objects": [{"id": "obj_1", "color": "blue"}],
                                       "places": [{"id": "paper_3", "label": "blue paper"}]})
    assert plain["items"]["obj_1"]["color"] == "blue" and plain["env"]["places"]["paper_3"]["color"] == "blue"


def test_one_sort_entry_becomes_every_object_into_its_colour_zone():
    goal, problem = openworld.resolve_goal([{"sort_by_color": True}], SC)
    assert problem is None and len(goal) == 16
    assert all(g[0] == "in" and g[2].split("_")[0] == SC["items"][g[1]]["color"] for g in goal)
    goal, _ = openworld.resolve_goal([{"sort_by_color": True}, {"object": "green_cube_2", "in": "red_zone"}], SC)
    assert ("in", "green_cube_2", "red_zone") in goal and len(goal) == 16


def test_sort_without_matching_places_says_why():
    sc = openworld.normalize_scene({"objects": [{"id": "red_cube", "color": "red"}],
                                    "places": [{"id": "box_1", "kind": "box"}]})
    goal, problem = openworld.resolve_goal([{"sort_by_color": True}], sc)
    assert goal is None and "no place has the colour" in problem


def test_four_tower_request_plans_with_one_goal_entry():
    m = FixedAnswer({"goal": [{"sort_by_color": True}]})
    r = openworld.run(instruction=REQ["instruction"], scene=REQ["scene"], vlm=m)
    assert r["status"] == "ok" and r["quality"]["optimal"] and r["quality"]["moves"] == 16 and m.calls == 1


def test_empty_model_goal_is_retried_and_explained():
    m = FixedAnswer({"goal": [], "constraints": {"no_table": True}, "evidence": {}})   # the answer seen on HEKATE
    r = openworld.run_until_ok(3, instruction=REQ["instruction"], scene=REQ["scene"], vlm=m)
    assert r["status"] == "clarify" and "empty goal" in r["message"]
    assert m.calls == 3 and len(r["attempts"]) == 3                 # asked again before giving up
