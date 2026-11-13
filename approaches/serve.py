"""HTTP server with a selectable planning strategy.

    STRATEGY=graph          uvicorn approaches.serve:app --port 8020     (default)
    STRATEGY=htn            uvicorn approaches.serve:app --port 8020
    STRATEGY=pddl:pyperplan uvicorn approaches.serve:app --port 8020
    STRATEGY=adaptive       uvicorn approaches.serve:app --port 8020
    STRATEGY=llm_program    uvicorn approaches.serve:app --port 8020     (needs generated/policy.py)
    also: astar, llm_heuristic, anytime (BUDGET_MS=5), cost_aware, cp_sat, partial_order

The robot controller always talks to the same three endpoints and receives the
same message format, whichever strategy is selected. Sessions, scene versions,
plan validation, feedback handling and replanning live in semplan.service and
are shared by all strategies.

    POST /v1/sessions/{sid}/plan       full scene message -> validated plan
    POST /v1/sessions/{sid}/scene      new observation (full or delta) -> kept | replanned
    POST /v1/sessions/{sid}/feedback   outcome of one step
    GET  /v1/sessions/{sid}            server-side state
    GET  /v1/metrics                   latency percentiles, case base, action statistics
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Request

from .base import Strategy
from semplan.scene import SceneError
from semplan.service import PlanningService, Stale


def make_strategy(name: str) -> Strategy | None:
    """Strategy by name. None selects the service's built-in graph planner."""
    if name in ("", "graph"):
        return None
    if name == "astar":
        from .symbolic import GraphStrategy
        return GraphStrategy(optimal=True)
    if name == "htn":
        from .htn import HtnStrategy
        return HtnStrategy()
    if name.startswith("pddl:"):
        from .symbolic import PddlStrategy
        return PddlStrategy(name.split(":", 1)[1])
    if name == "llm_program":
        from .llm_offline import ProgramStrategy
        return ProgramStrategy()
    if name == "llm_heuristic":
        from .llm_offline import HeuristicSearchStrategy
        return HeuristicSearchStrategy()
    if name == "adaptive":
        from .adaptive import AdaptivePlanner
        return AdaptivePlanner()
    if name == "anytime":
        from .anytime import AnytimeStrategy
        return AnytimeStrategy(float(os.environ.get("BUDGET_MS", "5")))
    if name == "cp_sat":
        from .cpsat import CpSatStrategy
        return CpSatStrategy()
    if name == "partial_order":
        from .partial_order import PartialOrderStrategy
        return PartialOrderStrategy()
    if name == "cost_aware":
        return None                                  # attached below: it needs the service's statistics
    raise ValueError(f"unknown STRATEGY {name!r}")


STRATEGY = os.environ.get("STRATEGY", "graph")
# The adaptive planner has its own case base; elsewhere the service's case base is used.
service = PlanningService(strategy=make_strategy(STRATEGY), use_cases=STRATEGY != "adaptive")
if STRATEGY == "cost_aware":
    from .cost_aware import CostAwareStrategy
    service.strategy = CostAwareStrategy(service.stats)   # plans with what step feedback has taught
app = FastAPI(title=f"vlm_pddl planning server ({STRATEGY})", version="0.1")


async def _call(fn, sid: str, request: Request):
    try:
        return fn(sid, await request.json())
    except Stale as e:
        raise HTTPException(409, str(e))
    except (SceneError, KeyError, TypeError, ValueError) as e:
        raise HTTPException(422, f"{type(e).__name__}: {e}")


@app.post("/v1/sessions/{sid}/plan")
async def plan(sid: str, request: Request):
    return await _call(service.plan, sid, request)


@app.post("/v1/sessions/{sid}/scene")
async def scene(sid: str, request: Request):
    return await _call(service.scene_update, sid, request)


@app.post("/v1/sessions/{sid}/feedback")
async def feedback(sid: str, request: Request):
    return await _call(service.feedback, sid, request)


@app.get("/v1/sessions/{sid}")
def state(sid: str):
    return service.state(sid)


@app.get("/v1/metrics")
def metrics():
    return {"strategy": STRATEGY, **service.metrics()}
