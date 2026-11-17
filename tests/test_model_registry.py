"""Model registry and loader options (no GPU: only the parts that decide how to load).
Run: python -m pytest tests/test_model_registry.py -q"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

yaml = pytest.importorskip("yaml")

from vlm import model_class_names, processor_kwargs, template_kwargs

ROOT = Path(__file__).resolve().parents[1]
MODELS = yaml.safe_load((ROOT / "models.yaml").read_text())["models"]


def test_every_model_has_an_id_and_known_options():
    for key, spec in MODELS.items():
        assert "/" in spec["hf"], key
        assert spec.get("auto_class", "image-text-to-text") in ("image-text-to-text", "multimodal-lm"), key
        assert spec.get("processor_size", "qwen") in ("qwen", "none"), key
        model_class_names(spec), processor_kwargs(spec, 1024), template_kwargs(spec)


def test_existing_models_load_exactly_as_before():
    spec = MODELS["qwen3-vl-8b"]
    assert model_class_names(spec)[0] == "AutoModelForImageTextToText"
    assert processor_kwargs(spec, 1024) == {"size": {"shortest_edge": 256 * 1024, "longest_edge": 1024 * 1024}}
    assert template_kwargs(spec) == {}


def test_fast_models_are_configured_to_stay_fast():
    for key in ("qwen3-vl-2b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-0.8b", "lfm2.5-vl-1.6b"):
        assert key in MODELS
        assert not MODELS[key].get("reasoning")
    assert template_kwargs(MODELS["qwen3.5-4b"]) == {"enable_thinking": False}     # thinks by default otherwise
    assert model_class_names(MODELS["qwen3.5-2b"])[0] == "AutoModelForMultimodalLM"
    lfm = processor_kwargs(MODELS["lfm2.5-vl-1.6b"], 512)
    assert "size" not in lfm and lfm["max_image_tokens"] == 256


def test_model_comparison_chart(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import run_report
    res = tmp_path / "results"
    res.mkdir()
    for model, secs in (("qwen3-vl-2b", 2.0), ("qwen3-vl-8b", 6.0)):
        (res / f"resolution_sweep_{model}.json").write_text(json.dumps({"model": model, "summary": {
            "low": {"runs": 9, "works": 6, "shortest": 6, "mean_support_acc": 0.8, "mean_seconds": secs,
                    "mean_image_tokens": 120},
            "high": {"runs": 9, "works": 8, "shortest": 8, "mean_support_acc": 0.9, "mean_seconds": secs * 2,
                     "mean_image_tokens": 900}}}))
    monkeypatch.setattr(run_report, "ROOT", tmp_path)
    assert run_report.chart_models(tmp_path) and (tmp_path / "models.png").exists()
