"""Scene understanding: many input forms -> one checked planner scene, then plan and replan.
Run: python -m pytest tests/test_scene_understanding.py -q"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from approaches.llm import MockLLM
from approaches.scene_understanding import understand_scene
from approaches.task_service import TaskService
from semplan.gen import random_scene_dict
from semplan.planner import plan
from semplan.scene import parse_scene

H = 0.04                                         # object height and footprint (m)
ZONE_Y = [-0.25, 0.0, 0.25]


def to_geometry(d: dict, rng: random.Random, noise: float = 0.003, units: str = "m") -> dict:
    """Render a planner scene as a depth camera would report it: positions only, no 'on'."""
    zones = {z["id"]: (0.30, ZONE_Y[i]) for i, z in enumerate(d["zones"])}
    by_id = {o["id"]: o for o in d["objects"]}
    stacks = []
    for o in d["objects"]:
        if o["on"] not in by_id:                  # bottom of a stack
            col, cur = [o["id"]], o["id"]
            while (up := next((x["id"] for x in d["objects"] if x["on"] == cur), None)):
                col.append(up)
                cur = up
            stacks.append((o["on"], col))
    used = {z: 0 for z in zones}
    table_i = 0
    objs = []
    for base, col in stacks:
        if base in zones:
            x, y = zones[base][0] + (-0.045, 0.0, 0.045)[used[base]], zones[base][1]
            used[base] += 1
        else:
            x, y = 0.55, -0.30 + 0.08 * table_i
            table_i += 1
        for k, oid in enumerate(col):
            o = by_id[oid]
            p = [x + rng.uniform(-noise, noise), y + rng.uniform(-noise, noise), H * k + H / 2]
            if units == "mm":
                p = [v * 1000 for v in p]
            objs.append({"id": oid, "shape": o["type"], "color": o["color"], "position": p,
                         "size": H * (1000 if units == "mm" else 1), "height": H * (1000 if units == "mm" else 1)})
    rng.shuffle(objs)
    return {"units": units, "objects": objs,
            "zones": [{"id": z["id"], "color": z["color"], "capacity": z["capacity"],
                       "x": zones[z["id"]][0] * (1000 if units == "mm" else 1),
                       "y": zones[z["id"]][1] * (1000 if units == "mm" else 1),
                       "size": 0.14 * (1000 if units == "mm" else 1)} for z in d["zones"]],
            "free_slots": d["free_slots"]}


def supports(scene: dict) -> dict:
    return {o["id"]: o.get("on", "table") for o in scene["objects"]}


# ---------------------------------------------------------------- formats

def test_planner_format_is_unchanged():
    rng = random.Random(1)
    for _ in range(200):
        d = random_scene_dict(rng, rng.randint(2, 8), n_colors=3, n_buffers=2)
        r = understand_scene(d)
        assert supports(r.scene) == supports(d)
        assert not r.uncertain
        a, b = parse_scene(d), parse_scene(r.scene)
        assert plan(a).moves == plan(b).moves


@pytest.mark.parametrize("units", ["m", "mm"])
def test_supports_recovered_from_positions_only(units):
    """No 'on' facts at all: the reader must rebuild the same scene from noisy positions."""
    rng = random.Random(7)
    checked = 0
    for _ in range(300):
        d = random_scene_dict(rng, rng.randint(2, 8), n_colors=3, n_buffers=2, zone_capacity=(1, 3))
        g = to_geometry(d, rng, units=units)
        r = understand_scene(g)
        assert supports(r.scene) == supports(d), (d, g)
        assert not r.uncertain
        assert plan(parse_scene(d)).moves == plan(parse_scene(r.scene)).moves
        checked += 1
    assert checked == 300


def test_robot_lab_export():
    raw = {"schema_version": 1, "source": "simulator_ground_truth", "units": "meters",
           "robot": {"held_object": None},
           "objects": [{"id": "object_1", "shape": "cube", "color": "red", "position": [0.45, 0.05, 0.02], "size": 0.04, "height": 0.04},
                       {"id": "object_2", "shape": "cylinder", "color": "blue", "position": [0.45, 0.05, 0.06], "size": 0.04, "height": 0.04},
                       {"id": "object_3", "shape": "cube", "color": "green", "position": [0.29, -0.20, 0.02], "size": 0.04, "height": 0.04}],
           "relations": [{"subject": "object_2", "relation": "on_top_of", "object": "object_1"}],
           "destinations": {"red_zone": {"x": .29, "y": -.21, "color": "red"},
                            "blue_zone": {"x": .29, "y": 0, "color": "blue"},
                            "temporary_zone": {"x": .12, "y": .27, "color": "temporary"}}}
    r = understand_scene(raw)
    assert supports(r.scene) == {"object_1": "table", "object_2": "object_1", "object_3": "red_zone"}
    assert r.scene["free_slots"] == ["temporary_zone"]
    assert r.source_format == "robot_lab"


def test_detections_without_ids_in_mm():
    raw = {"detections": [{"label": "red cube", "xyz": [450, 50, 20], "score": .93},
                          {"label": "blue cylinder", "xyz": [452, 48, 62], "score": .88}],
           "zones": [{"id": "blue_paper", "x": 300, "y": 0, "size": 140}]}
    r = understand_scene(raw)
    assert supports(r.scene) == {"red_cube_1": "table", "blue_cylinder_1": "red_cube_1"}
    assert {o["id"]: o["type"] for o in r.scene["objects"]} == {"red_cube_1": "cube", "blue_cylinder_1": "cylinder"}
    assert r.scene["zones"][0]["color"] == "blue"
    assert any("millimetres" in a for a in r.assumptions)


def test_open_world_vlm_format():
    raw = {"objects": [{"id": "red_cylinder", "label": "red cylinder"}, {"id": "green_block", "label": "green block"}],
           "places": [{"id": "green_zone", "kind": "zone"}],
           "support": {"red_cylinder": "green_block", "green_block": "unknown"}}
    r = understand_scene(raw)
    assert supports(r.scene)["red_cylinder"] == "green_block"
    assert "green_block" in r.uncertain                 # support unknown -> frozen, not guessed


def test_contradictions_are_frozen_not_guessed():
    raw = {"objects": [{"id": "a", "type": "cube", "color": "red", "on": "c"},
                       {"id": "b", "type": "cube", "color": "blue", "on": "c"},
                       {"id": "c", "type": "cube", "color": "green", "on": "d"},
                       {"id": "d", "type": "cube", "color": "green"},
                       {"id": "e", "type": "cube", "color": "red", "on": "ghost"},
                       {"id": "f", "type": "cube", "color": "blue"}],
           "zones": [{"id": "red_zone", "color": "red", "capacity": 2},
                     {"id": "blue_zone", "color": "blue", "capacity": 2}], "free_slots": []}
    r = understand_scene(raw)
    assert set(r.uncertain) == {"a", "b", "c", "d", "e"}
    res = plan(parse_scene(r.scene))
    assert {m.obj for m in res.moves} == {"f"}          # only the object nobody doubts is moved


def test_position_contradicts_relation():
    raw = {"objects": [{"id": "a", "shape": "cube", "color": "red", "position": [0.5, 0.0, 0.02]},
                       {"id": "b", "shape": "cube", "color": "blue", "position": [0.5, 0.2, 0.02]}],
           "relations": [{"subject": "a", "relation": "on_top_of", "object": "b"}]}
    r = understand_scene(raw)
    assert "a" in r.uncertain


# ---------------------------------------------------------------- whole server path

def lab_scene(**moved):
    pos = {"red": [0.55, -0.1, 0.02], "blue": [0.55, -0.1, 0.06], "green": [0.55, 0.1, 0.02], "yellow": [0.45, 0.1, 0.02]}
    pos.update(moved)
    shape = {"red": "cube", "blue": "cylinder", "green": "cube", "yellow": "cube"}
    return {"units": "meters", "objects": [{"id": f"{c}_1", "shape": shape[c], "color": c, "position": p,
                                           "size": 0.04, "height": 0.04} for c, p in pos.items()],
            "destinations": {"red_zone": {"x": .29, "y": -.21, "color": "red"},
                             "blue_zone": {"x": .29, "y": 0, "color": "blue"},
                             "temporary_zone": {"x": .12, "y": .27, "color": "temporary"}}}


def moves(resp):
    return [(s["object"], s["destination"]) for s in resp["steps"] if s["action"] != "PICK"]


def test_instruction_on_simulator_scene_then_replan():
    svc = TaskService(memory_path=None)
    r = svc.task("s", {"scene": lab_scene(),
                       "instruction": "Pick red and place on green, then pick blue and place on yellow."})
    assert r["status"] == "ok", r
    assert r["understanding"]["source"] == "rules"
    assert moves(r)[-2:] == [("red_1", "green_1"), ("blue_1", "yellow_1")] or \
        set(moves(r)) >= {("red_1", "green_1"), ("blue_1", "yellow_1")}
    # someone puts the blue cylinder on the yellow cube by hand before the robot starts
    r2 = svc.scene_update("s", lab_scene(blue=[0.45, 0.1, 0.06]))
    assert r2["decision"] == "replanned"
    assert moves(r2) == [("red_1", "green_1")]
    assert r2["scene_understanding"]["source_format"] in ("generic", "robot_lab")


def test_feedback_with_observation_in_another_format():
    svc = TaskService(memory_path=None)
    r = svc.task("s", {"scene": lab_scene(), "instruction": "put the red cube on the green cube"})
    first = r["steps"][0]
    assert first["action"] == "PICK" and first["object"] == "blue_1"   # blue covers red
    # the pick succeeded; the laptop reports the scene with blue in the gripper
    obs = lab_scene()
    obs["robot"] = {"held_object": "blue_1"}
    r2 = svc.feedback("s", {"plan_id": r["plan_id"], "step": first["step"], "status": "success", "scene": obs})
    assert r2["status"] == "ok" and r2["steps"][0]["object"] == "blue_1"


def test_images_go_through_the_model_then_the_same_reader():
    class FakeVLM:
        def generate(self, paths, prompt, max_visual_tokens=1024, **kw):
            assert "Describe the CURRENT scene" in prompt
            return {"text": json.dumps({
                "objects": [{"id": "red_cube", "label": "red cube"}, {"id": "green_cube", "label": "green cube"}],
                "places": [{"id": "red_zone", "kind": "zone"}, {"id": "green_zone", "kind": "zone"}],
                "support": {"red_cube": "table", "green_cube": "red_cube"}})}
    llm = MockLLM([])
    llm.model = FakeVLM()
    svc = TaskService(llm, memory_path=None)
    r = svc.task("s", {"images": ["view0.jpg"], "instruction": "sort by colour"})
    assert r["status"] == "ok"
    assert r["scene_understanding"]["source_format"] == "images"
    assert moves(r)[0][0] == "green_cube"                # unstack before sorting


def test_unidentified_colour_asks_for_a_view_then_plans():
    svc = TaskService(memory_path=None)
    scene = lab_scene()
    scene["objects"][0]["color"] = None                  # the camera could not see red_1's colour
    r = svc.task("s", {"scene": scene, "instruction": "put the red cube on the green cube"})
    assert r["status"] == "needs_observation"
    assert r["info_requests"] == [{"action": "REQUEST_NEW_VIEW", "target": "red_1",
                                   "reason": "the instruction may refer to this unidentified object"}]
    r2 = svc.scene_update("s", lab_scene())              # new view: colour visible now
    assert r2["status"] == "ok"
    assert moves(r2)[-1] == ("red_1", "green_1")


def test_http_accepts_simulator_json():
    from fastapi.testclient import TestClient
    from approaches.task_server import create_app
    c = TestClient(create_app(TaskService(memory_path=None)))
    r = c.post("/v1/sessions/arm/task", json={"scene": lab_scene(), "instruction": "sort by colour"})
    assert r.status_code == 200, r.text
    assert r.json()["scene_understanding"]["source_format"] in ("generic", "robot_lab")
    bad = c.post("/v1/sessions/arm2/task", json={"scene": {"objects": []}, "instruction": "sort"})
    assert bad.status_code == 422


# ---------------------------------------------------------------- robot side renumbers objects

def renamed(sc, mapping):
    sc = dict(sc)
    sc["objects"] = [dict(o, id=mapping.get(o["id"], o["id"])) for o in sc["objects"]]
    if isinstance(sc.get("robot"), dict) and sc["robot"].get("held_object"):
        sc["robot"] = {"held_object": mapping.get(sc["robot"]["held_object"], sc["robot"]["held_object"])}
    return sc


def test_new_ids_every_observation_keep_the_task():
    svc = TaskService(memory_path=None)
    svc.task("s", {"scene": lab_scene(), "instruction": "put the red cube on the green cube"})
    r = svc.scene_update("s", renamed(lab_scene(), {"red_1": "obj_7", "green_1": "obj_3", "blue_1": "obj_9", "yellow_1": "obj_2"}))
    assert r["status"] == "ok"
    assert moves(r) == [("obj_9", "red_zone"), ("obj_7", "obj_3")]      # answers in the newest ids
    r = svc.scene_update("s", renamed(lab_scene(), {"red_1": "a", "green_1": "b", "blue_1": "c", "yellow_1": "d"}))
    assert moves(r) == [("c", "red_zone"), ("a", "b")]


def test_object_of_the_task_missing_asks_to_look():
    svc = TaskService(memory_path=None)
    svc.task("s", {"scene": lab_scene(), "instruction": "put the red cube on the green cube"})
    gone = renamed(lab_scene(), {"red_1": "a", "blue_1": "c", "yellow_1": "d"})
    gone["objects"] = [o for o in gone["objects"] if o["id"] != "green_1"]
    r = svc.scene_update("s", gone)
    assert r["status"] == "needs_observation" and r["steps"] == []


def test_two_same_cubes_told_apart_by_position():
    from approaches.scene_understanding import match_ids
    old = [{"id": "x1", "type": "cube", "color": "red", "position": [0.5, -0.1, 0.02]},
           {"id": "x2", "type": "cube", "color": "red", "position": [0.5, 0.1, 0.02]}]
    new = [{"id": "n2", "type": "cube", "color": "red", "position": [0.503, 0.098, 0.02]},
           {"id": "n1", "type": "cube", "color": "red", "position": [0.499, -0.102, 0.02]}]
    assert match_ids(old, new) == {"x1": "n1", "x2": "n2"}


def test_held_object_is_not_swapped_for_an_unidentified_one():
    from approaches.scene_understanding import match_ids
    old = [{"id": "blue", "type": "cube", "color": "blue", "position": [0.5, 0.0, 0.02]},
           {"id": "unk", "type": "cube", "color": None, "position": [0.52, 0.0, 0.02]}]
    new = [{"id": "h", "type": "cube", "color": "blue", "on": "gripper"},
           {"id": "u", "type": "cube", "color": None, "position": [0.52, 0.0, 0.02]}]
    assert match_ids(old, new) == {"blue": "h", "unk": "u"}
    old2 = [dict(new[0], id="h0"), dict(new[1], id="u0")]
    new2 = [dict(new[0], id="h1", color=None), dict(new[1], id="u1")]
    assert match_ids(old2, new2)["h0"] == "h1"            # one gripper: held before = held now
