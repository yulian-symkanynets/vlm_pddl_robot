"""Tests for general tasks: specification, planner, understanding, service, HTTP.
Run: python -m pytest tests/test_task_server.py -q"""
import copy
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from approaches.llm import MockLLM
from approaches.mock import planner_oracle
from approaches.model_bench import make_case, same_task
from approaches.task_service import TaskService, check_contract, from_openworld
from approaches.understand import ExampleMemory, LLMInterpreter, RuleInterpreter, TaskInterpreter
from semplan.planner import plan_astar
from semplan.scene import Move, SceneError, parse_scene, validate
from semplan.service import PlanningService
from semplan.task_planner import plan_task

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


def with_task(task, scene=SCENE):
    return parse_scene(dict(copy.deepcopy(scene), task=copy.deepcopy(task)))


def solve(task, scene=SCENE):
    s = with_task(task, scene)
    r = plan_task(s)
    assert r.status == "ok", r.reason
    assert validate(s, r.moves).ok
    return s, r.moves


# ---------------------------------------------------------------- task specification + planner

def test_tower_in_a_given_order():
    s, moves = solve({"goal": [{"stack": ["green_cube_1", "red_cube_1", "blue_cylinder_1"], "in": "blue_zone"}]})
    final = validate(s, moves).final
    assert final.on["red_cube_1"] == "green_cube_1" and final.on["blue_cylinder_1"] == "red_cube_1"
    assert final.base("blue_cylinder_1") == "blue_zone" and len(moves) == 4


def test_constraints_change_the_plan_and_are_enforced_by_the_validator():
    swap = {"goal": [{"object": "red_cube_1", "on": "blue_cylinder_1"}]}
    _, free = solve(swap)
    s, strict = solve(dict(swap, constraints={"no_temporary": True, "never_on": [["*", "zone"]]}))
    assert all(m.dst in s.objects for m in strict)                       # only object-to-object moves
    assert "forbids temporary slots" in validate(s, [Move("blue_cylinder_1", "tmp_1")], False).reason
    s, moves = solve({"goal": [{"sort_by_color": True}], "constraints": {"keep": ["green_prism_1"]}})
    assert all(m.obj != "green_prism_1" for m in moves)
    assert "must not be moved" in validate(s, [Move("green_prism_1", "green_zone")], False).reason
    s, moves = solve({"goal": [{"sort_by_color": True}], "constraints": {"max_stack_height": 1}})
    assert all(m.dst in s.locs for m in moves)
    assert len(free) == 2


def test_impossible_and_contradictory_tasks_are_reported():
    for bad, word in (({"goal": [{"object": "red_cube_1", "on": "blue_cylinder_1"},
                                 {"object": "blue_cylinder_1", "on": "red_cube_1"}]}, "circular"),
                      ({"goal": [{"object": "red_cube_1", "on": "green_prism_1"}]}, "prism"),
                      ({"goal": [{"object": "yellow_ball", "in": "red_zone"}]}, "unknown object"),
                      ({"goal": [{"object": "red_cube_1", "in": "red_zone"},
                                 {"object": "red_cube_1", "in": "blue_zone"}]}, "two different goal zones"),
                      ({"goal": []}, "no goal")):
        with pytest.raises(SceneError, match=word):
            with_task(bad)
    s = with_task({"goal": [{"object": "red_cube_1", "in": "red_zone"}],
                   "constraints": {"keep": ["blue_cylinder_1"]}})       # the cylinder sits on the cube
    r = plan_task(s)
    assert r.status == "infeasible" and "red_cube_1 in red_zone" in r.reason


def test_sort_only_tasks_use_the_fast_planner_and_match_optimal_length():
    s = with_task({"goal": [{"sort_by_color": True}]})
    assert not s.general
    assert len(plan_task(s).moves) == len(plan_astar(s).moves)


def test_collision_risk_forces_the_blocker_to_move_first():
    d = copy.deepcopy(SCENE)
    d["objects"][2]["blocked_by"] = ["blue_cylinder_1"]                  # tall stack next to the green cube
    s = parse_scene(d)
    assert s.general and "collision risk" in validate(s, [Move("green_cube_1", "green_zone")], False).reason
    moves = plan_task(s).moves
    order = [m.obj for m in moves]
    assert validate(s, moves).ok and order.index("blue_cylinder_1") < order.index("green_cube_1")


def test_random_tasks_plans_are_valid_for_the_intended_task():
    rng = random.Random(3)
    n = 0
    while n < 60:
        case = make_case(rng)
        if case is None:
            continue
        n += 1
        s = with_task(case["task"], case["scene"])
        r = plan_task(s)
        assert r.status == "ok" and validate(s, r.moves).ok


# ---------------------------------------------------------------- understanding

@pytest.mark.parametrize("instruction,expected", [
    ("Sort the objects by colour.", {"goal": [{"sort_by_color": True}]}),
    ("put the red cube on the green cube", {"goal": [{"object": "red_cube_1", "on": "green_cube_1"}]}),
    ("Put the red cube on the green cube, but don't put anything on the table",
     {"goal": [{"object": "red_cube_1", "on": "green_cube_1"}], "constraints": {"no_temporary": True}}),
    ("stack the green cube, the red cube and the blue cylinder from bottom to top in the blue zone",
     {"goal": [{"stack": ["green_cube_1", "red_cube_1", "blue_cylinder_1"], "in": "blue_zone"}]}),
    ("move the blue cylinder to the red zone and do not touch the green prism",
     {"goal": [{"object": "blue_cylinder_1", "in": "red_zone"}], "constraints": {"keep": ["green_prism_1"]}}),
    ("rearrange them without using the temporary slots so that the green cube is under the red cube",
     {"goal": [{"object": "red_cube_1", "on": "green_cube_1"}], "constraints": {"no_temporary": True}}),
    ("sort by colour and never put anything on the red cube",
     {"goal": [{"sort_by_color": True}], "constraints": {"never_on": [["*", "red_cube_1"]]}}),
])
def test_rule_parser_understands_common_phrasings(instruction, expected):
    task, problem = RuleInterpreter().interpret(instruction, parse_scene(SCENE))
    assert task is not None, problem
    assert same_task(SCENE, task, expected)


def test_rule_parser_asks_instead_of_guessing():
    rules, scene = RuleInterpreter(), parse_scene(SCENE)
    task, q = rules.interpret("put the cube on the cylinder", scene)
    assert task is None and "which one" in q
    task, q = rules.interpret("put the yellow ball in the red zone", scene)
    assert task is None and "no yellow ball" in q
    assert rules.interpret("make it look nice", scene)[0] is None


def test_model_answer_is_validated_and_corrected():
    scene = parse_scene(SCENE)
    llm = MockLLM(['{"goal": [{"object": "red_cube", "on": "green_cube_1"}]}',
                   'Here you go: {"goal": [{"object": "red_cube_1", "on": "green_cube_1"}]}'])
    u = LLMInterpreter(llm, ExampleMemory(None)).interpret("Polož červenú kocku na zelenú kocku", scene)
    assert u.task == {"goal": [{"object": "red_cube_1", "on": "green_cube_1"}]} and u.rounds == 2
    assert u.errors == ["unknown object 'red_cube'"] and "was rejected" in llm.prompts[1]
    first = llm.prompts[0]
    assert "red_cube_1: red cube" in first and '"on": "red_cube_1"' not in first      # no current arrangement
    assert "Examples of earlier instructions" in first and u.examples_used > 0       # retrieved examples
    ask = LLMInterpreter(MockLLM(['{"goal": [], "question": "Which cube?"}'])).interpret("move it", scene)
    assert ask.task is None and ask.question == "Which cube?"
    hopeless = LLMInterpreter(MockLLM(["no idea"]), max_rounds=2).interpret("x", scene)
    assert hopeless.task is None and hopeless.llm_calls == 2


def test_chain_uses_the_model_only_when_rules_fail_and_caches_the_result():
    llm = MockLLM(['{"goal": [{"object": "red_cube_1", "on": "green_cube_1"}]}'])
    chain = TaskInterpreter(llm, ExampleMemory(None))
    scene = parse_scene(SCENE)
    assert chain.understand("put the red cube on the green cube", scene).source == "rules" and llm.calls == 0
    u = chain.understand("the red cube belongs right above the green one", scene)
    assert u.source == "llm" and llm.calls == 1
    again = chain.understand("The red cube belongs right above the green one", scene)
    assert again.source == "cache" and llm.calls == 1 and again.task == u.task


# ---------------------------------------------------------------- service

def test_service_plans_replans_and_remembers(tmp_path):
    svc = TaskService(None, "symbolic", memory_path=tmp_path / "examples.json")
    r = svc.task("s", {"scene": SCENE, "instruction": "put the red cube on the green cube, but don't use the table"})
    assert r["status"] == "ok" and r["source"].startswith("task_") and r["understanding"]["source"] == "rules"
    assert check_contract(r["steps"], svc.kb) is None
    # a person puts the prism... nothing can sit on a prism, so they move the green cube into the red zone
    r2 = svc.scene_update("s", {"scene_version": 2, "delta": {"objects": [{"id": "green_cube_1", "on": "red_zone"}]}})
    assert r2["task"] == r["task"] and r2["status"] == "ok"
    assert all(s.get("destination") != "tmp_1" for s in r2["steps"])     # constraint survives replanning
    before = len(svc.memory.items)
    while r2["steps"]:                                                   # execute to the end
        r2 = svc.feedback("s", {"plan_id": r2["plan_id"], "step": r2["steps"][0]["step"], "status": "success"})
    assert r2["status"] == "done" and len(svc.memory.items) == before + 1
    assert json.loads((tmp_path / "examples.json").read_text())[0]["task"] == r["task"]


def test_service_clarifies_and_rejects_bad_structured_tasks():
    svc = TaskService(None, "symbolic", memory_path=None)
    r = svc.task("s", {"scene": SCENE, "instruction": "put the cube on the cylinder"})
    assert r["status"] == "clarify" and r["steps"] == [] and "which one" in r["question"]
    with pytest.raises(SceneError):
        svc.task("s", {"scene": SCENE, "task": {"goal": [{"object": "red_cube_1", "on": "green_prism_1"}]}})
    default = svc.task("d", {"scene": SCENE})
    assert default["status"] == "ok" and default["understanding"]["source"] == "default"


def test_vlm_plan_mode_checks_the_models_plan_against_the_task():
    llm = MockLLM(["MOVE blue_cylinder_1 tmp_1\nMOVE red_cube_1 green_cube_1",           # breaks the constraint
                   "MOVE blue_cylinder_1 blue_zone\nMOVE red_cube_1 green_cube_1"])
    svc = TaskService(llm, "vlm_plan", order="rules", memory_path=None)
    r = svc.task("s", {"scene": SCENE, "instruction": "put the red cube on the green cube and don't use the table"})
    assert r["status"] == "ok" and r["llm_calls"] == {"understand": 0, "plan": 2}
    assert "forbids temporary slots" in llm.prompts[1]
    bad = TaskService(MockLLM(["MOVE red_cube_1 green_cube_1"]), "vlm_plan", order="rules", memory_path=None)
    assert bad.task("s", {"scene": SCENE, "instruction": "put the red cube on the green cube"})["status"] == "infeasible"
    oracle = TaskService(MockLLM(planner_oracle()), "vlm_plan", order="rules", memory_path=None)
    assert oracle.task("s", {"scene": SCENE, "instruction": "sort by colour"})["status"] == "ok"


def test_contract_check_and_openworld_conversion():
    kb = TaskService(None, "symbolic", memory_path=None).kb
    assert "not an operation" in check_contract([{"step": 0, "action": "THROW", "object": "a"}], kb)
    assert "lacks destination" in check_contract([{"step": 0, "action": "PLACE", "object": "a"}], kb)
    msg = from_openworld({"objects": [{"id": "red_cylinder", "label": "red cylinder"},
                                      {"id": "thing", "label": "something"}],
                          "places": [{"id": "red_paper", "kind": "zone", "capacity": 2}],
                          "support": {"red_cylinder": "table", "thing": "unknown"}})
    s = parse_scene(msg)
    assert s.objects["red_cylinder"].type == "cylinder" and s.zone_of("red_cylinder") == "red_paper"
    assert s.deferred()["thing"] == "unidentified"


def test_http_api():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from approaches.task_server import create_app
    c = TestClient(create_app(TaskService(None, "symbolic", memory_path=None)))
    r = c.post("/v1/sessions/a/task", json={"scene": SCENE, "instruction":
               "stack the green cube, the red cube and the blue cylinder from bottom to top"})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "ok" and body["task"]["goal"][0]["stack"][0] == "green_cube_1"
    assert c.post("/v1/sessions/a/feedback", json={"plan_id": body["plan_id"], "step": 0,
                                                   "status": "success"}).status_code == 200
    assert c.post("/v1/sessions/a/feedback", json={"plan_id": "old", "step": 0, "status": "success"}).status_code == 409
    assert c.post("/v1/sessions/b/task", json={"scene": SCENE, "instruction": "put the cube on the cylinder"}
                  ).json()["status"] == "clarify"
    bad = c.post("/v1/sessions/c/task", json={"scene": SCENE, "task": {"goal": [{"object": "nope", "in": "red_zone"}]}})
    assert bad.status_code == 422 and "unknown object" in bad.json()["detail"]
    assert {o["name"] for o in c.get("/v1/knowledge").json()["operations"]} >= {"PICK", "PLACE", "REQUEST_NEW_VIEW"}
    assert c.get("/v1/metrics").json()["understanding"]["rules"] == 1
    assert c.get("/health").json() == {"status": "ok", "mode": "symbolic", "model": None}
    assert c.post("/v1/perceive", json={"images_b64": []}).status_code == 503
