"""Tests for the semantic communication experiment and the GNN policy.
Run: python -m pytest tests/test_semcom_gnn.py -q"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from approaches.semcom import (cases, classify, compact, json_full, json_min, packed, plan_for,
                               task_relevant, unpack)
from semplan.scene import Move, parse_scene, validate

CASES = cases(120, seed=5)


def test_smaller_representations_are_really_smaller():
    for c in CASES:
        full, small, binary = len(json_full(c["msg"])), len(json_min(c["msg"])), len(packed(c["msg"])[0])
        assert binary < small < full


def test_minimal_json_and_packed_give_the_identical_plan():
    for c in CASES:
        msg = c["msg"]
        _, ref = plan_for(msg)
        _, a = plan_for(json.loads(json_min(msg)))
        payload, back = packed(msg)
        _, b = plan_for(unpack(payload, msg))
        b = [Move(back.get(m.obj, m.obj), back.get(m.dst, m.dst)) for m in b]
        assert a == ref and b == ref


def test_packed_keeps_uncertainty_within_one_step_of_quantisation():
    msg = CASES[0]["msg"]
    payload, back = packed(msg)
    got = {back[o["id"]]: o for o in unpack(payload, msg)["objects"]}
    for o in msg["objects"]:
        assert abs(got[o["id"]]["confidence"] - o["confidence"]) <= 1 / 255
        assert got[o["id"]]["type"] == o["type"] and got[o["id"]]["color"] == o["color"]


def test_task_relevant_subset_never_yields_a_plan_that_fails_on_the_real_scene():
    fewer = 0
    for c in CASES:
        msg = c["msg"]
        _, ref = plan_for(msg)
        reduced = task_relevant(msg)
        fewer += len(reduced["objects"]) < len(msg["objects"])
        _, moves = plan_for(reduced)
        if moves is not None:                               # None would trigger the second round
            assert validate(parse_scene(json.loads(json.dumps(msg))), moves).ok
            assert classify(msg, ref, moves) in ("identical", "equivalent", "valid_longer")
    assert fewer > 10


def test_classification_of_results():
    msg = CASES[0]["msg"]
    _, ref = plan_for(msg)
    assert classify(msg, ref, ref) == "identical" and classify(msg, ref, None) == "failed"
    assert classify(msg, ref, ref[:-1]) == "failed"          # does not achieve the task


def test_ground_truth_of_the_photographed_scenes_loads_and_plans():
    pytest.importorskip("yaml")
    from approaches.semcom import real_scene_messages
    scenes = real_scene_messages()
    assert len(scenes) >= 5
    for sc in scenes:
        _, moves = plan_for(sc["msg"])
        assert moves, sc["scene_id"]
        assert len(packed(sc["msg"])[0]) < 40


def test_gnn_policy_trains_and_never_returns_an_invalid_plan():
    pytest.importorskip("torch")
    from approaches import run
    from approaches.gnn_policy import GnnPolicyStrategy, encode, graph_dataset, train_gnn
    from approaches.symbolic import GraphStrategy
    from semplan.gen import random_scene
    from semplan.planner import quick_infeasible
    data = graph_dataset(60, seed=1)
    x, edges, idx = encode(parse_scene(CASES[-1]["msg"]))
    assert x.shape[0] == len(idx) and set(edges) == {"on", "carries", "goal", "wanted_by"}
    model, acc = train_gnn(data, epochs=4)
    assert acc > 0.4
    rng, n = random.Random(2), 0
    with_fallback = GnnPolicyStrategy(model, GraphStrategy())
    while n < 40:
        s = random_scene(rng, rng.randint(3, 6), n_colors=3, n_buffers=2)
        if quick_infeasible(s) or run(GraphStrategy(), s).status != "ok":
            continue
        n += 1
        assert run(GnnPolicyStrategy(model), s).status in ("ok", "no_plan")
        assert run(with_fallback, s).status == "ok"
