"""General goals: the model writes a formula, the server grounds it, searches and verifies.
Run: python -m pytest tests/test_goal_formula.py -q"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import formula_plan
import goal_logic
import openworld

ROOT = Path(__file__).resolve().parents[1]
SCENE = json.loads((ROOT / "examples" / "requests" / "15_four_towers_sort.json").read_text())["scene"]
SC = openworld.normalize_scene(SCENE)

STACK = {"formula": {"and": [
    {"forall": {"x": "object"}, "body": {"or": [
        {"on_table": ["x"]},
        {"exists": {"y": "object"}, "where": {"same_color": ["x", "y"]}, "body": {"on": ["x", "y"]}}]}},
    {"forall": {"x": "object", "y": "object"},
     "where": {"and": [{"same_color": ["x", "y"]}, {"not": {"is": ["x", "y"]}}]},
     "body": {"not": {"and": [{"on_table": ["x"]}, {"on_table": ["y"]}]}}}]}}
CUBES = {"formula": {"forall": {"x": "object"}, "where": {"shape": ["x", "cube"]},
                     "body": {"in": ["x", "red_zone"]}}}


class FixedAnswer:
    key = "fixed"

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), 0

    def generate(self, image_paths, prompt, *a, **kw):
        ans = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        return {"text": json.dumps(ans), "generate_s": 0.0}


def stacks(scene, plan):
    sup = {o["id"]: o["on"] for o in scene["objects"]}
    for p in plan:
        if p["action"] == "place-on-table":
            sup[p["args"][0]] = "table"
        elif p["action"] == "place-on-item":
            sup[p["args"][0]] = p["args"][1]
        elif p["action"] == "place-in":
            sup[p["args"][0]] = p["args"][2]
    return sup


def test_stack_by_colour_written_as_a_formula_is_solved_fast():
    m = FixedAnswer({"goal": [STACK]})
    r = openworld.run(instruction="every stack one colour", scene=SCENE, vlm=m)
    assert r["status"] == "ok" and r["planner"] == "goal-search" and r["verified"]
    assert r["timings"]["plan_s"] < 8
    sup = stacks(SCENE, r["plan"])
    col = {o["id"]: o["color"] for o in SCENE["objects"]}
    assert all(sup[x] == "table" or col[sup[x]] == col[x] for x in sup)          # same colour below
    bases = [x for x in sup if sup[x] == "table"]
    assert len({col[x] for x in bases}) == len(bases) == 4                        # one stack per colour
    assert r["quality"]["optimal"] is False and "not proven shortest" in r["quality"]["note"]
    assert "same colour" in r["goal"][0]["in_words"]


def test_formula_with_a_shape_filter_and_a_zone():
    r = openworld.run(instruction="cubes into red", scene=SCENE, vlm=FixedAnswer({"goal": [CUBES]}))
    assert r["status"] == "ok" and r["verified"]
    sup = stacks(SCENE, r["plan"])
    cubes = [o["id"] for o in SCENE["objects"] if "cube" in o["id"]]
    assert cubes and all(sup[c] == "red_zone" for c in cubes)


def test_bad_formulas_get_exact_messages_and_a_retry():
    bad = [{"formula": {"sparkle": ["red_cube_1"]}}, {"formula": {"on": ["red_cube_1", "nowhere"]}},
           {"formula": {"in": ["red_cube_1", "green_bar_1"]}}, {"formula": {"on": ["red_cube_1"]}},
           {"formula": {"forall": {"red_cube_1": "object"}, "body": {"on_table": ["red_cube_1"]}}}]
    for entry in bad:
        _, problem = openworld.resolve_goal([entry], SC)
        assert problem and problem.startswith("the goal formula is invalid"), (entry, problem)
    m = FixedAnswer({"goal": [bad[0]]}, {"goal": [bad[0]]}, {"goal": [CUBES]})
    r = openworld.run_until_ok(3, instruction="cubes into red", scene=SCENE, vlm=m)
    assert r["status"] == "ok" and m.calls == 3                      # a slip is retried, not reported


def test_contradictory_formula_is_refused_before_searching():
    f = {"formula": {"forall": {"x": "object"}, "body": {"and": [{"color": ["x", "red"]}, {"on_table": ["x"]}]}}}
    _, problem = openworld.resolve_goal([f], SC)
    assert "can never be true" in problem


def test_already_true_goal_needs_no_moves():
    f = {"formula": {"on": ["blue_block_1", "green_bar_1"]}}
    r = openworld.run(scene=SCENE, goal=[f])
    assert r["status"] == "ok" and r["plan"] == [] and r["quality"]["optimal"]


def test_constraints_are_respected(monkeypatch):
    monkeypatch.setattr(formula_plan, 'LIMIT_S', 1.5)             # impossible goals are searched until this
    r = openworld.run(scene=SCENE, goal=[{"formula": {"on_table": ["green_cube_2"]}}], constraints={"no_table": True})
    assert r["status"] == "no_plan"                                   # it must reach the table, but none may be used
    r = openworld.run(scene=SCENE, goal=[{"formula": {"in": ["red_block_1", "red_zone"]}}],
                      constraints={"keep": ["red_block_1"]})
    assert r["status"] == "no_plan"                                   # kept in place, so it cannot go to the zone
    r = openworld.run(scene=SCENE, goal=[{"formula": {"in": ["red_block_1", "red_zone"]}}])
    assert r["status"] == "ok" and r["quality"]["moves"] == 1


def test_replan_keeps_a_formula_goal():
    r = openworld.run(scene=SCENE, goal=[CUBES])
    first = r["plan"][1]                                              # the first place action
    new = json.loads(json.dumps(SCENE))
    for o in new["objects"]:
        if o["id"] == first["args"][0]:
            o["on"] = first["args"][2] if first["action"] == "place-in" else first["args"][-1]
    r2 = openworld.replan(r, new)
    assert r2["status"] == "ok" and r2["goal"][0]["formula"] == CUBES["formula"]


def test_describe_mentions_every_part():
    text = goal_logic.describe(CUBES["formula"])
    assert "for every x" in text and "is a cube" in text and "is in red_zone" in text
