"""Every scenario of examples/scenarios.json, without a model (rule parser, or the structured
task when the rules cannot read the sentence). Run: python -m pytest tests/test_scenarios.py -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from approaches.run_scenarios import load, run_one
from approaches.task_service import TaskService

DATA = load()


@pytest.mark.parametrize("sc", DATA["scenarios"], ids=[s["name"] for s in DATA["scenarios"]])
def test_scenario(sc):
    r = run_one(sc, DATA, lambda: TaskService(memory_path=None, use_cache=False))
    assert r["ok"], f"{r['problems']} | understood by {r['understood_by']} | task {r['task']} | moves {r['moves']}"
