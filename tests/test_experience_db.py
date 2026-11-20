"""Experience database and run graphs (stand-in model, no GPU).
Run: python -m pytest tests/test_experience_db.py -q"""
import base64
import copy
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import openworld
from experience_db import ExperienceDB

ROOT = Path(__file__).resolve().parents[1]
PHOTO = ROOT / "images" / "scene1.jpg"
SCENE = openworld.DEMO_SCENE
ASK = "put the red cylinder in the red zone"


class CountingModel:
    """The rule-based stand-in model, counting calls and keeping the prompts."""
    key = "counting"

    def __init__(self):
        self.inner, self.calls, self.prompts = openworld.RuleMock(), {"ow_perceive": 0, "ow_goal": 0}, []

    def generate(self, image_paths, prompt, max_visual_tokens=1024, max_new_tokens=None, sample=False, context=None):
        self.calls[context["mode"]] += 1
        self.prompts.append(prompt)
        return self.inner.generate(image_paths, prompt, max_visual_tokens, max_new_tokens, sample, context)


@pytest.fixture
def db(tmp_path):
    return ExperienceDB(tmp_path / "exp.sqlite", model_key="test")


def test_same_request_twice_is_answered_from_the_database(db):
    m = CountingModel()
    a = openworld.run(instruction=ASK, scene=SCENE, vlm=m, memory=db)
    b = openworld.run(instruction=ASK, scene=SCENE, vlm=m, memory=db)
    assert a["status"] == b["status"] == "ok"
    assert a["memory"] == {"goal": "miss", "plan": "miss"} and b["memory"] == {"goal": "hit", "plan": "hit"}
    assert m.calls["ow_goal"] == 1                                    # the second request asked no model
    assert a["plan"] == b["plan"] and a["quality"] == b["quality"] and b["verified"]
    assert b["timings"]["goal_s"] == 0.0


def test_moved_objects_reuse_the_goal_but_plan_again(db):
    m = CountingModel()
    openworld.run(instruction=ASK, scene=SCENE, vlm=m, memory=db)
    moved = copy.deepcopy(SCENE)
    moved["support"] = {**moved["support"], "red_cylinder": "table", "green_block": "table"}
    r = openworld.run(instruction=ASK, scene=moved, vlm=m, memory=db)
    assert r["status"] == "ok" and r["memory"] == {"goal": "hit", "plan": "miss"}
    assert m.calls["ow_goal"] == 1


def test_other_words_other_model_or_other_objects_miss(db, tmp_path):
    m = CountingModel()
    openworld.run(instruction=ASK, scene=SCENE, vlm=m, memory=db)
    assert openworld.run(instruction="put the blue cylinder in the blue zone", scene=SCENE, vlm=m,
                         memory=db)["memory"]["goal"] == "miss"
    other_model = ExperienceDB(db.path, model_key="another model")
    assert openworld.run(instruction=ASK, scene=SCENE, vlm=m, memory=other_model)["memory"]["goal"] == "miss"
    fewer = copy.deepcopy(SCENE)
    fewer["objects"] = fewer["objects"][:3]
    fewer["support"].pop("blue_cylinder")
    fewer["support"]["green_block"] = "table"
    assert openworld.run(instruction=ASK, scene=fewer, vlm=m, memory=db)["memory"]["goal"] == "miss"


def test_only_successful_requests_are_stored_and_retries_do_not_read(db):
    m = CountingModel()
    r = openworld.run(instruction="put the purple ball in the red zone", scene=SCENE, vlm=m, memory=db)
    assert r["status"] != "ok" and db.stats()["stored"]["goal"]["entries"] == 0
    openworld.run(instruction=ASK, scene=SCENE, vlm=m, memory=db)
    retry = openworld.run(instruction=ASK, scene=SCENE, vlm=m, memory=db, sample=True)
    assert retry["memory"]["goal"] == "miss" and m.calls["ow_goal"] == 3


def test_same_image_skips_perception(db, tmp_path):
    m = CountingModel()
    img = tmp_path / "view.jpg"
    img.write_bytes(PHOTO.read_bytes())
    a = openworld.run(instruction=ASK, images=[img], vlm=m, memory=db)
    b = openworld.run(instruction=ASK, images=[img], vlm=m, memory=db)
    assert a["status"] == b["status"] == "ok"
    assert b["memory"] == {"scene": "hit", "goal": "hit", "plan": "hit"}
    assert m.calls == {"ow_perceive": 1, "ow_goal": 1}
    other = tmp_path / "other.jpg"
    other.write_bytes(PHOTO.read_bytes() + b"\x00")                   # one byte different
    assert openworld.run(instruction=ASK, images=[other], vlm=m, memory=db)["memory"]["scene"] == "miss"


def test_similar_instructions_become_examples_only_when_switched_on(tmp_path):
    db = ExperienceDB(tmp_path / "e.sqlite", model_key="test", examples=2)
    m = CountingModel()
    openworld.run(instruction=ASK, scene=SCENE, vlm=m, memory=db)
    r = openworld.run(instruction="put the blue cylinder in the red zone", scene=SCENE, vlm=m, memory=db)
    assert r["memory"].get("examples") == 1
    assert "Earlier instructions in other scenes" in m.prompts[-1] and ASK in m.prompts[-1]
    off = ExperienceDB(tmp_path / "e.sqlite", model_key="test", examples=0)
    openworld.run(instruction="put the yellow block in the red zone", scene=SCENE, vlm=m, memory=off)
    assert "Earlier instructions" not in m.prompts[-1]


def test_server_endpoints_and_dashboard(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("matplotlib")
    from fastapi.testclient import TestClient
    import server
    logs = tmp_path / "requests"
    db = ExperienceDB(tmp_path / "exp.sqlite", model_key="test")
    c = TestClient(server.create_app(CountingModel(), 1024, log_dir=logs, memory=db))
    b64 = base64.b64encode(PHOTO.read_bytes()).decode()
    for body in ({"instruction": ASK, "scene": SCENE}, {"instruction": ASK, "scene": SCENE},
                 {"instruction": ASK, "images_b64": [b64], "resolution": "low"},
                 {"instruction": ASK, "images_b64": [b64], "resolution": "low"}):
        assert c.post("/plan", json=body).json()["status"] == "ok"
    stats = c.get("/memory").json()
    assert stats["stored"]["goal"]["entries"] >= 1 and stats["this_run"]["goal"]["hits"] >= 2
    page = c.get("/dashboard").text
    for chart in ("stages.png", "by_type.png", "memory.png", "resolution.png", "outcomes.png"):
        assert chart in page
        assert c.get(f"/dashboard/files/{chart}").status_code == 200
    assert c.get("/dashboard/files/../../secret.png").status_code == 404
    assert c.post("/memory/clear", params={"kind": "goal"}).json()["deleted"] >= 1
    assert c.get("/memory").json()["stored"]["goal"]["entries"] == 0


def test_repeated_ids_and_colour_shape_descriptions():
    dup = {"objects": [{"id": "red_block", "on": "table"}, {"id": "blue_block", "on": "red_block"},
                       {"id": "red_block", "on": "blue_block"}]}
    d = openworld.normalize_scene(dup)
    assert d["support"] == {"red_block": "table", "blue_block": "red_block", "red_block_2": "blue_block"}
    both = {"objects": [{"id": "red_block", "on": "table"}, {"id": "red_block", "on": "table"},
                        {"id": "blue_block", "on": "red_block"}]}
    assert openworld.normalize_scene(both)["unknown"] == ["blue_block"]      # really ambiguous: ask
    s = json.loads((ROOT / "examples" / "scenes" / "four_towers_2026-11-20.json").read_text())
    sc = openworld.normalize_scene(s)
    assert len(sc["items"]) == 16 and sc["items"]["green_bar_1"]["label"] == "green bar"
    goal = [{"object": o["id"], "in": f"{o['color']}_zone"} for o in s["objects"]]
    r = openworld.run(scene=s, goal=goal)
    assert r["status"] == "ok" and r["quality"]["optimal"] and r["quality"]["moves"] == 16


def test_fixed_places_are_added_to_what_the_camera_saw(tmp_path):
    m = CountingModel()
    img = tmp_path / "view.jpg"
    img.write_bytes(PHOTO.read_bytes())
    zones = [{"id": "purple_zone", "kind": "zone", "capacity": 2}]
    r = openworld.run(instruction="put the red cylinder in the purple zone", images=[img], vlm=m, places=zones)
    assert "purple_zone" in r["scene"]["places"][-1]["id"] or any(
        p["id"] == "purple_zone" for p in r["scene"]["places"])
