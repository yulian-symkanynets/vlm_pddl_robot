"""Tests for approaches/ (no GPU, no model). Run: python -m pytest tests/test_approaches.py -q"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from approaches import run, scenarios as S
from approaches.active_perception import decide
from approaches.adaptive import AdaptivePlanner
from approaches.behavior_tree import run_tree
from approaches.htn import HtnStrategy
from approaches.llm import MockLLM, extract_code, extract_json, extract_moves
from approaches.llm_offline import ProgramStrategy, load_function, rollout, synthesize_policy
from approaches.llm_online import LLMDirectStrategy, LLMModuloStrategy, LLMPlusPStrategy
from approaches.mock import POLICY_V1, POLICY_V2, POLICY_V3, goal_oracle, planner_oracle
from approaches.replanning import FullReplan, PrefixRepair, ValidateThenReplan
from approaches.symbolic import CaseBasedStrategy, GraphStrategy
from semplan.gen import random_scene
from semplan.planner import quick_infeasible
from semplan.scene import Move, SceneError, parse_scene, validate
from semplan.sim import SimRobot

BLOCKED = parse_scene(S.BLOCKED)


def solvable(seed, count, lo=3, hi=6):
    rng, out = random.Random(seed), []
    while len(out) < count:
        s = random_scene(rng, rng.randint(lo, hi), n_colors=3, n_buffers=rng.choice([1, 2]))
        if not quick_infeasible(s) and run(GraphStrategy(), s).status == "ok":
            out.append(s)
    return out


@pytest.mark.parametrize("strategy", [GraphStrategy(), GraphStrategy(optimal=True), HtnStrategy(),
                                      CaseBasedStrategy()], ids=lambda s: s.name)
def test_symbolic_strategies_solve_the_blocked_scene(strategy):
    o = run(strategy, BLOCKED)
    assert o.status == "ok" and len(o.moves) == 4
    assert o.moves[-1].dst in ("blue_zone", "red_cube_1")


def test_pddl_strategy_gives_a_plan_the_shared_validator_accepts():
    pytest.importorskip("unified_planning")
    from approaches.symbolic import PddlStrategy
    strategy = PddlStrategy("pyperplan")
    assert run(strategy, BLOCKED).status == "ok"
    assert set(strategy.last_timing) == {"generate_ms", "parse_ms", "solve_ms"}


def test_every_symbolic_strategy_is_valid_or_silent_on_random_scenes():
    for s in solvable(1, 80):
        for strategy in (GraphStrategy(), HtnStrategy()):
            assert run(strategy, s).status in ("ok", "no_plan")     # never an invalid plan


def test_case_base_reuses_and_never_returns_an_invalid_plan():
    cb = CaseBasedStrategy()
    assert "miss" in run(cb, BLOCKED).source
    assert run(cb, BLOCKED).source == "case"


def test_direct_llm_plan_is_caught_and_modulo_loop_repairs_it():
    bad = run(LLMDirectStrategy(MockLLM(planner_oracle(error_rate=1.0))), BLOCKED)
    assert bad.status == "invalid" and "not clear" in bad.reason
    fixed = run(LLMModuloStrategy(MockLLM(planner_oracle(error_rate=1.0))), BLOCKED)
    assert fixed.status == "ok" and fixed.llm_calls == 2
    hopeless = run(LLMModuloStrategy(MockLLM(["MOVE red_cube_1 red_zone"]), max_rounds=3), BLOCKED)
    assert hopeless.status == "no_plan" and hopeless.llm_calls == 3


def test_llm_plus_p_translates_only_the_goal_and_rejects_unknown_names():
    o = run(LLMPlusPStrategy(MockLLM(goal_oracle), instruction="put the red cube in the green zone"),
            BLOCKED)
    assert o.status == "ok" and o.llm_calls == 1 and o.moves[-1] == Move("red_cube_1", "green_zone")
    assert run(LLMPlusPStrategy(MockLLM(goal_oracle)), BLOCKED).llm_calls == 0     # no instruction
    made_up = MockLLM(['{"goal": {"yellow_ball": "red_zone"}}'])
    o = run(LLMPlusPStrategy(made_up, instruction="move the yellow ball"), BLOCKED)
    assert o.status == "no_plan" and "unknown object" in o.reason


def test_goal_message_forms():
    for goal in ({"object": "red_cube_1", "destination": "green_zone"},
                 [{"object": "red_cube_1", "destination": "green_zone"}],
                 {"red_cube_1": "green_zone"}):
        s = parse_scene(dict(S.BLOCKED, goal=goal))
        assert s.goals() == {"red_cube_1": "green_zone"}
    with pytest.raises(SceneError):
        parse_scene(dict(S.BLOCKED, goal={"red_cube_1": "tmp_1"}))


def test_policy_synthesis_improves_with_feedback_and_program_runs_without_model():
    train = solvable(2, 30, 3, 5)
    code, log = synthesize_policy(MockLLM([POLICY_V1, POLICY_V2, POLICY_V3]), train,
                                  accept=0.8, save_as=None)
    solved = [l["solved"] for l in log]
    assert code is not None and solved[0] < solved[1] < solved[2]
    for s in solvable(3, 60, 3, 7):
        assert run(ProgramStrategy(code), s).status in ("ok", "no_plan")


def test_generated_code_cannot_import_and_bad_code_is_reported():
    with pytest.raises(Exception):
        load_function("import os\ndef policy(v):\n    return None", "policy")
    crash = load_function("def policy(v):\n    return v.nothing()", "policy")
    assert "raised AttributeError" in rollout(crash, BLOCKED)[1]
    loop = load_function("def policy(v):\n    return ('red_prism_1', 'tmp_1') if v.on('red_prism_1') "
                         "!= 'tmp_1' else ('red_prism_1', 'blue_zone')", "policy")
    assert "endless loop" in rollout(loop, BLOCKED)[1]


def test_active_perception_acts_on_agreement_and_asks_when_it_matters():
    probs = {"unknown_1": {"blue": 0.55, "green": 0.45}}
    d = decide(parse_scene(S.OCCLUDED), probs)
    assert d["action"] == "EXECUTE" and d["moves"] == [Move("green_cube_1", "green_zone")]
    lone = parse_scene({"objects": [{"id": "u", "type": "cube", "color": None, "confidence": 0.3}],
                        "zones": S.ZONES, "free_slots": ["t"]})
    assert decide(lone, {"u": {"red": 0.5, "blue": 0.5}}) ["action"] == "REQUEST_NEW_VIEW"


def test_replanning_methods_agree_on_validity_and_validate_keeps_a_good_plan():
    before, after = parse_scene(S.CHANGE_BEFORE), parse_scene(S.CHANGE_AFTER)
    plan = run(GraphStrategy(), before).moves
    remaining = [m for m in plan if m.obj != "blue_cylinder_1"]
    for method in (FullReplan(), ValidateThenReplan(), PrefixRepair()):
        r = method.replan(after, remaining)
        assert validate(after, r.moves).ok and r.based_on_version == 2
        assert r.moves[0] == Move("red_cube_1", "red_zone") and r.decision != "kept"
    unchanged = parse_scene(dict(S.CHANGE_BEFORE, scene_version=2))
    assert ValidateThenReplan().replan(unchanged, plan).decision == "kept"


def test_behavior_tree_recovers_from_disturbance_failures_and_hidden_objects():
    rng = random.Random(5)
    for i, s in enumerate(solvable(6, 40, 3, 6)):
        hidden = set(rng.sample(sorted(s.objects), 1)) if i % 2 else set()
        robot = SimRobot(s, rng, p_pick_fail=0.15, hidden=hidden)
        out = run_tree(robot, disturb=lambda t, r=robot: r.human_disturbance()
                       if rng.random() < 0.1 and r.truth.held() is None else None)
        assert out["goal"], out["trace"][-5:]


def test_adaptive_planner_uses_the_model_only_when_symbolic_stages_cannot_answer():
    llm = MockLLM(planner_oracle())
    adaptive = AdaptivePlanner(llm_fallback=LLMModuloStrategy(llm))
    for s in solvable(7, 60):
        assert run(adaptive, s).status == "ok"
    assert llm.calls == 0 and adaptive.answered.get("graph", 0) > 0


def test_output_parsers():
    assert extract_moves("1. MOVE(a -> red_zone)\nMOVE b, tmp_1\nnoise") == [("a", "red_zone"), ("b", "tmp_1")]
    assert extract_json('<think>{"x":1}</think> here: {"goal": {"a": "z"}} done') == {"goal": {"a": "z"}}
    assert extract_code("text\n```python\ndef f():\n    pass\n```\nmore") == "def f():\n    pass"


@pytest.mark.parametrize("name", ["graph", "htn", "adaptive"])
def test_server_gives_the_same_contract_for_every_strategy(name, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    import importlib
    from fastapi.testclient import TestClient
    monkeypatch.setenv("STRATEGY", name)
    import approaches.serve as serve
    client = TestClient(importlib.reload(serve).app)
    r = client.post("/v1/sessions/t/plan", json=S.CHANGE_BEFORE).json()
    assert r["status"] == "ok" and [s["action"] for s in r["steps"]] == ["PICK", "PLACE"] * 3
    first = r["steps"][0]
    ok = client.post("/v1/sessions/t/feedback", json={"plan_id": r["plan_id"], "step": first["step"],
                                                       "status": "success"})
    assert ok.status_code == 200
    stale = client.post("/v1/sessions/t/feedback", json={"plan_id": "old", "step": 0, "status": "success"})
    assert stale.status_code == 409
    r2 = client.post("/v1/sessions/t2/plan", json=S.CHANGE_BEFORE).json()
    upd = client.post("/v1/sessions/t2/scene", json=S.CHANGE_AFTER).json()
    assert upd["decision"] == "replanned" and upd["based_on_version"] == 2
    assert upd["steps"][0]["object"] == "red_cube_1"
