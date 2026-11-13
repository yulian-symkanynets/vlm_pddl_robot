"""Tests for the additional strategies. Run: python -m pytest tests/test_more_strategies.py -q"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from approaches import run, scenarios as S
from approaches.anytime import AnytimeStrategy, eliminate_redundant
from approaches.calibration import calibrate, error_bound, synthetic_log
from approaches.contingent import ContingentStrategy, build_tree, count_leaves, follow
from approaches.cost_aware import CostAwareStrategy, expected_attempts
from approaches.partial_order import PartialOrderStrategy, deorder
from approaches.symbolic import GraphStrategy
from semplan.cases import ActionStats
from semplan.gen import random_scene
from semplan.planner import plan_astar, quick_infeasible
from semplan.scene import Move, dump_scene, parse_scene, validate

BLOCKED = parse_scene(S.BLOCKED)


def solvable(seed, count, lo=3, hi=7):
    rng, out = random.Random(seed), []
    while len(out) < count:
        s = random_scene(rng, rng.randint(lo, hi), n_colors=3, n_buffers=rng.choice([0, 1, 2]))
        if not quick_infeasible(s) and run(GraphStrategy(), s).status == "ok":
            out.append(s)
    return out


def test_every_ordering_allowed_by_the_partial_order_is_a_valid_plan():
    rng = random.Random(1)
    for s in solvable(1, 120):
        plan = deorder(s, run(GraphStrategy(), s).moves)
        for _ in range(4):
            assert validate(s, plan.linearize(cost=lambda a, b: rng.random())).ok


def test_partial_order_structure_on_known_scenes():
    chain = deorder(BLOCKED, run(GraphStrategy(), BLOCKED).moves)
    assert chain.edges() == [(0, 1), (1, 2), (2, 3)] and chain.flexibility() == 0.0
    free = parse_scene(S.CHANGE_BEFORE)
    plan = deorder(free, run(GraphStrategy(), free).moves)
    assert plan.edges() == [] and plan.ready(set()) == [0, 1, 2] and plan.flexibility() == 1.0
    assert plan.affected_by({"green_cube_1"}) == {1}
    assert run(PartialOrderStrategy(), free).status == "ok"


def test_cost_aware_avoids_an_unreliable_slot_when_an_alternative_exists():
    stats = ActionStats()
    stats.counts[("PLACE", "cylinder", "buffer")] = [30, 70]          # the slot is a bad place
    o = run(CostAwareStrategy(stats), BLOCKED)
    assert o.status == "ok" and Move("blue_cylinder_1", "tmp_1") not in o.moves
    graph = run(GraphStrategy(), BLOCKED).moves
    assert expected_attempts(BLOCKED, o.moves, stats) < expected_attempts(BLOCKED, graph, stats)
    neutral = run(CostAwareStrategy(ActionStats()), BLOCKED)           # no statistics: shortest plan
    assert len(neutral.moves) == len(plan_astar(BLOCKED).moves)


def test_contingent_tree_is_valid_for_every_hypothesis():
    rng = random.Random(3)
    for s in solvable(3, 60, 4, 6):
        d, truth = dump_scene(s), {}
        hidden = rng.sample(sorted(s.objects), 2)
        for o in d["objects"]:
            if o["id"] in hidden:
                truth[o["id"]] = o["color"]
                o["color"], o["confidence"] = None, 0.3
        tree = build_tree(parse_scene(d), {})
        # Some colour hypotheses describe an unsolvable scene; the branch for the
        # true colours must always be a complete valid plan.
        assert validate(s, follow(tree, truth)).ok
    tree = build_tree(parse_scene(S.OCCLUDED), {"unknown_1": {"blue": 0.55, "green": 0.45}})
    assert tree["moves"] == [Move("green_cube_1", "green_zone")] and tree["observe"] == "unknown_1"
    assert count_leaves(tree) == 2 and list(tree["branches"]) == ["blue", "green"]
    o = run(ContingentStrategy({"unknown_1": {"blue": 0.55, "green": 0.45}}), parse_scene(S.OCCLUDED))
    assert o.status == "partial" and o.info_requests[0]["target"] == "unknown_1"


def test_anytime_is_always_valid_and_never_worse_with_more_time():
    for s in solvable(4, 60, 6, 9):
        quick, slow = run(AnytimeStrategy(0), s), run(AnytimeStrategy(30), s)
        assert quick.status == "ok" and slow.status == "ok"
        assert len(slow.moves) <= len(quick.moves)
    detour = [Move("blue_cylinder_1", "green_zone"), Move("blue_cylinder_1", "tmp_1"),
              Move("red_cube_1", "red_zone"), Move("red_prism_1", "red_cube_1"),
              Move("blue_cylinder_1", "blue_zone")]
    assert validate(BLOCKED, detour).ok and len(eliminate_redundant(BLOCKED, detour)) == 4


def test_cp_sat_is_optimal_and_respects_side_constraints():
    pytest.importorskip("ortools")
    from approaches.cpsat import CpSatStrategy
    for s in solvable(5, 12, 3, 5):
        o = run(CpSatStrategy(), s)
        assert o.status == "ok" and len(o.moves) == len(plan_astar(s).moves)
    o = run(CpSatStrategy(avoid=["tmp_1"]), BLOCKED)
    assert o.status == "ok" and all(m.dst != "tmp_1" for m in o.moves)
    assert run(CpSatStrategy(move_once={"blue_cylinder_1"}), BLOCKED).status == "ok"
    impossible = run(CpSatStrategy(forbid_on=[("red_prism_1", "red_cube_1")]), BLOCKED)
    assert impossible.status == "no_plan" and "INFEASIBLE" in impossible.reason


def test_calibration_bound_and_threshold():
    assert error_bound(0, 100, 0.1) == pytest.approx(0.0228, abs=0.001)     # 1 - 0.1 ** (1/100)
    assert error_bound(5, 100, 0.1) > 0.05 and error_bound(0, 0, 0.1) == 1.0
    r = calibrate(synthetic_log(3000, seed=2), alpha=0.05, delta=0.1)
    assert r["threshold"] is not None and r["error_bound"] <= 0.05
    assert r["empirical_error"] < r["error_bound"]
    assert calibrate(synthetic_log(30, seed=2), alpha=0.01)["threshold"] is None   # too little data
