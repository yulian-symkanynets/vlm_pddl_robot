"""Per-request image resolution, crop and auto escalation (stand-in model, no GPU).
Run: python -m pytest tests/test_image_resolution.py -q"""
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import image_resolution
import openworld

ROOT = Path(__file__).resolve().parents[1]
PHOTO = ROOT / "images" / "scene1.jpg"


class EyesightModel:
    """Answers with the demo scene, but below `sharp_px` it cannot tell what stands on what."""
    key = "eyesight"

    def __init__(self, sharp_px=700):
        self.sharp_px, self.seen = sharp_px, []

    def generate(self, image_paths, prompt, max_visual_tokens=1024, max_new_tokens=None, sample=False, context=None):
        from PIL import Image
        if context and context.get("mode") == "ow_perceive":
            with Image.open(image_paths[0]) as im:
                side = max(im.size)
            self.seen.append((side, max_visual_tokens))
            scene = json.loads(json.dumps(openworld.DEMO_SCENE))
            if side < self.sharp_px:
                scene["support"]["red_cylinder"] = "unknown"
            return {"text": json.dumps(scene), "generate_s": side / 1000}
        return openworld.RuleMock().generate(image_paths, prompt, max_visual_tokens, max_new_tokens, sample, context)


def test_levels_and_budgets(tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(PHOTO.read_bytes())
    for level, side in (("low", 448), ("medium", 768), ("high", 1280), (640, 640)):
        paths, info = image_resolution.prepare([img], level, None, 1024)
        assert max(info["images"][0]["width"], info["images"][0]["height"]) == side
        assert info["images"][0]["bytes"] < img.stat().st_size
    _, low = image_resolution.prepare([img], "low", None, 1024)
    _, high = image_resolution.prepare([img], "high", None, 1024)
    assert low["images"][0]["approx_tokens"] < high["images"][0]["approx_tokens"]
    paths, info = image_resolution.prepare([img], "full", None, 1024)
    assert paths == [img] and info["max_visual_tokens"] == 1024
    with pytest.raises(ValueError):
        image_resolution.prepare([img], "tiny", None, 1024)
    with pytest.raises(ValueError):
        image_resolution.prepare([img], "low", [0.5, 0, 0.4, 1], 1024)


def test_crop_keeps_the_work_area(tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(PHOTO.read_bytes())
    _, full = image_resolution.prepare([img], "medium", None, 1024)
    _, crop = image_resolution.prepare([img], "medium", [0.25, 0.25, 0.75, 0.75], 1024)
    assert crop["images"][0]["bytes"] < full["images"][0]["bytes"] or \
        crop["images"][0]["width"] <= full["images"][0]["width"]


def client(model):
    from fastapi.testclient import TestClient
    import server
    return TestClient(server.create_app(model, 1024, log_dir=Path(__import__("tempfile").mkdtemp())))


def test_auto_climbs_only_as_far_as_needed():
    pytest.importorskip("fastapi")
    m = EyesightModel(sharp_px=700)
    b64 = base64.b64encode(PHOTO.read_bytes()).decode()
    r = client(m).post("/plan", json={"instruction": "put the red cylinder in the red zone",
                                      "images_b64": [b64], "resolution": "auto"}).json()
    assert r["status"] == "ok", r.get("message")
    tried = r["resolution"]["tried"]
    assert [t["resolution"] for t in tried] == ["low", "medium"] and r["resolution"]["used"] == "medium"
    assert tried[0]["not_reliable_because"] == "an object's support is unknown"
    assert [s for s, _ in m.seen] == [448, 768]


def test_fixed_level_is_used_as_given_and_low_is_enough_for_a_sharp_model():
    pytest.importorskip("fastapi")
    b64 = base64.b64encode(PHOTO.read_bytes()).decode()
    m = EyesightModel(sharp_px=300)
    r = client(m).post("/plan", json={"instruction": "put the red cylinder in the red zone",
                                      "images_b64": [b64], "resolution": "auto"}).json()
    assert r["status"] == "ok" and r["resolution"]["used"] == "low" and len(m.seen) == 1
    m = EyesightModel(sharp_px=300)
    r = client(m).post("/plan", json={"instruction": "put the red cylinder in the red zone",
                                      "images_b64": [b64], "resolution": 512}).json()
    assert r["resolution"]["used"] == "512px" and m.seen == [(512, 256)]
    bad = client(m).post("/plan", json={"instruction": "x", "images_b64": [b64], "resolution": "huge"})
    assert bad.status_code == 422


def test_json_requests_are_not_affected():
    pytest.importorskip("fastapi")
    r = client(EyesightModel()).post("/plan", json={"instruction": "put the red cylinder in the red zone",
                                                    "scene": openworld.DEMO_SCENE, "resolution": "low"}).json()
    assert r["status"] == "ok" and "resolution" not in r


def test_perceive_endpoint_with_auto():
    pytest.importorskip("fastapi")
    m = EyesightModel(sharp_px=1000)
    with PHOTO.open("rb") as f:
        r = client(m).post("/perceive", files={"files": ("a.jpg", f, "image/jpeg")},
                           data={"resolution": "auto"}).json()
    assert [t["resolution"] for t in r["resolution"]["tried"]] == ["low", "medium", "high"]
