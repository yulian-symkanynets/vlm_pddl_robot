"""Common interface for all planning strategies."""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # repo root on the path

from semplan.scene import Move, Scene, validate  # noqa: E402


@dataclass
class Proposal:
    """What a strategy returns. `moves=None` means the strategy has no plan."""
    moves: list[Move] | None
    reason: str = ""
    source: str = ""                      # which internal stage produced it
    llm_calls: int = 0                    # model calls made for THIS request
    info_requests: list[dict] = field(default_factory=list)
    complete: bool = True                 # False: a deliberately partial plan (safe prefix)
    scene: Scene | None = None            # set when the strategy changed the goal (e.g. from
                                          # an instruction); the plan is validated against it


class Strategy(Protocol):
    name: str

    def propose(self, scene: Scene) -> Proposal: ...


@dataclass
class Outcome:
    strategy: str
    status: str                           # ok | partial | no_plan | invalid | needs_observation
    moves: list[Move]
    ms: float                             # wall time of propose(), validation excluded
    validate_ms: float
    reason: str = ""
    source: str = ""
    llm_calls: int = 0
    info_requests: list[dict] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.status in ("ok", "partial")


def run(strategy: Strategy, scene: Scene) -> Outcome:
    """Call a strategy and judge its answer with the shared validator.

    The validator replays the plan on the scene: every precondition must hold and
    the goal must be reached. A plan from any source that fails here is "invalid"
    and must not be executed.
    """
    t0 = time.perf_counter()
    p = strategy.propose(scene)
    ms = (time.perf_counter() - t0) * 1e3
    if p.moves is None:
        status = "needs_observation" if p.info_requests else "no_plan"
        return Outcome(strategy.name, status, [], ms, 0.0, p.reason, p.source, p.llm_calls,
                       p.info_requests)
    scene = p.scene or scene
    t1 = time.perf_counter()
    v = validate(scene, p.moves, require_goal=p.complete)
    vms = (time.perf_counter() - t1) * 1e3
    if not v.ok:
        return Outcome(strategy.name, "invalid", p.moves, ms, vms, v.reason, p.source,
                       p.llm_calls, p.info_requests)
    status = "partial" if scene.deferred() or p.info_requests or not p.complete else "ok"
    return Outcome(strategy.name, status, p.moves, ms, vms, p.reason, p.source, p.llm_calls,
                   p.info_requests)
