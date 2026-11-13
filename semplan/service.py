"""Stateful planning service: sessions, scene versions, feedback, replanning.

Framework-free on purpose: server.py is a thin FastAPI wrapper, and tests,
benchmarks and the simulator call this class directly.

Rules the service enforces
  * The last *observed* scene is the state. Predicted effects are applied only
    for steps the controller reports as successful, and are overwritten by the
    next observation.
  * Every plan is tied to the scene version it was computed from and gets a new
    plan_id. Feedback for an old plan_id or an old scene version is rejected.
  * One lock per session serialises plan / update / feedback. Planning takes well
    under a millisecond, so a scene update can never arrive "during" planning;
    it simply waits its turn and then supersedes the previous plan.
  * Nothing is returned for execution unless it passed validate() on the current scene.
"""
from __future__ import annotations

import copy
import threading
import time
import uuid
from dataclasses import dataclass, field

from .cases import ActionStats, Case, CaseBase, canonicalize
from .planner import plan as run_planner
from .task_planner import plan_task
from .scene import (HELD, Move, Scene, SceneError, apply, parse_scene, to_contract, validate)

REQUEST_FOR = {
    "unidentified": "REQUEST_NEW_VIEW",
    "low_identity_confidence": "REQUEST_NEW_VIEW",
    "occluded": "CHANGE_VIEWPOINT",
    "pose_unverified": "VERIFY_SCENE",
}


class Stale(Exception):
    """The message refers to an outdated plan or scene version (HTTP 409)."""


@dataclass
class Session:
    id: str
    raw: dict = field(default_factory=dict)          # last scene message (for deltas)
    scene: Scene | None = None
    moves: list[Move] = field(default_factory=list)  # remaining moves of the current plan
    contract: list[dict] = field(default_factory=list)
    cursor: int = 0                                  # next contract step expected
    plan_id: str = ""
    plan_version: int = -1
    case: Case | None = None
    retries: dict[str, int] = field(default_factory=dict)
    quarantined: set[str] = field(default_factory=set)
    awaiting_observation: bool = False
    state_source: str = "observed"                   # "observed" | "predicted"
    history: list[dict] = field(default_factory=list)
    replans: int = 0
    retrieval: str = ""                              # outcome of the last case lookup
    lock: threading.Lock = field(default_factory=threading.Lock)


class PlanningService:
    def __init__(self, cases: CaseBase | None = None, use_cases: bool = True,
                 max_retries: int = 1, min_retry_p: float = 0.3, astar_budget: int = 200_000,
                 astar_ms: float | None = 50.0, strategy=None, task_ms: float = 300.0):
        self.cases = cases if cases is not None else CaseBase()
        self.use_cases = use_cases
        self.stats = ActionStats()
        self.max_retries = max_retries
        self.min_retry_p = min_retry_p
        self.astar_budget = astar_budget
        self.astar_ms = astar_ms                     # wall-clock cap for the A* fallback
        # Optional planning strategy: any object with propose(scene) -> proposal that has
        # .moves (list of Move or None), .reason and .source. Replaces the built-in planner;
        # sessions, versions, validation, case base and feedback handling stay the same.
        self.strategy = strategy
        self.task_ms = task_ms                       # time budget of the general task planner
        self.sessions: dict[str, Session] = {}
        self.log: list[dict] = []                    # one latency record per request
        self._guard = threading.Lock()

    # ------------------------------------------------------------ helpers
    def _session(self, sid: str) -> Session:
        with self._guard:
            if sid not in self.sessions:
                self.sessions[sid] = Session(sid)
            return self.sessions[sid]

    def _load(self, ses: Session, raw: dict) -> Scene:
        raw = copy.deepcopy(raw)
        for o in raw.get("objects", []):             # quarantined objects stay untouchable
            if o["id"] in ses.quarantined:
                o["reachable"] = False
        return parse_scene(raw)

    def _make_plan(self, ses: Session, decision: str, t: dict) -> dict:
        """Case retrieval -> fallback planner -> validation -> retention."""
        scene = ses.scene
        t0 = time.perf_counter()
        source, moves, status, reason, case = "", [], "ok", "", None
        canon = None
        ses.retrieval = "off"
        # General tasks (non-zone goals, constraints, collision blockers) are not covered by
        # the structural case key or the fast sorter: they go to the general task planner.
        general = scene.general
        if self.use_cases and not general:
            canon = canonicalize(scene)
            r = self.cases.retrieve(scene, canon)
            ses.retrieval = r.outcome + (f": {r.reason}" if r.outcome == "rejected" else "")
            if r.outcome == "hit":
                source, moves, case = "case", r.moves, r.case
            elif r.outcome == "known_infeasible":
                source, status, reason = "case", "infeasible", r.reason
        t["retrieve_ms"] = (time.perf_counter() - t0) * 1e3
        t1 = time.perf_counter()
        if not source and general and self.strategy is None:
            res = plan_task(scene, self.task_ms)
            source, moves, reason = res.planner, res.moves, res.reason
            if res.status == "infeasible":
                status, moves = "infeasible", []
        elif not source and self.strategy is not None:
            p = self.strategy.propose(scene)
            source, moves, reason = p.source or self.strategy.name, p.moves or [], p.reason
            if p.moves is None:
                status = "infeasible"
        elif not source:
            res = run_planner(scene, astar_budget=self.astar_budget, astar_ms=self.astar_ms)
            source, moves, reason = res.planner, res.moves, res.reason
            if res.status == "infeasible":
                status, moves = "infeasible", []
                if self.use_cases and not general:
                    self.cases.retain_infeasible(
                        canon, reason, proven=reason != "search budget exhausted")
        t["plan_ms"] = (time.perf_counter() - t1) * 1e3
        t2 = time.perf_counter()
        if status != "infeasible":
            v = validate(scene, moves)
            if not v.ok:                              # never hand out an unverified plan
                status, reason, moves = "rejected", v.reason, []
            elif source != "case" and self.use_cases and moves and not general:
                case = self.cases.retain(scene, moves, source, canon)
        t["validate_ms"] = (time.perf_counter() - t2) * 1e3
        deferred = scene.deferred()
        if status == "ok" and deferred:
            status = "partial"
        if status in ("ok", "partial") and not moves and not deferred:
            status = "done"
        ses.moves, ses.case = moves, case
        ses.contract = to_contract(scene, moves)
        ses.cursor = 0
        ses.plan_id = uuid.uuid4().hex[:12]
        ses.plan_version = scene.version
        ses.awaiting_observation = False
        return self._response(ses, status, decision, source, reason)

    def _response(self, ses: Session, status: str, decision: str, source: str,
                  reason: str = "", extra_steps: list[dict] | None = None) -> dict:
        scene = ses.scene
        deferred = scene.deferred() if scene else {}
        requests = [{"action": REQUEST_FOR[why], "target": o, "reason": why}
                    for o, why in sorted(deferred.items()) if why in REQUEST_FOR]
        requests += [{"action": "VERIFY_SCENE", "target": o, "reason": "repeated_failure"}
                     for o in sorted(ses.quarantined) if scene and o in scene.objects]
        if status == "infeasible":
            requests.append({"action": "VERIFY_SCENE", "target": None,
                             "reason": "no valid plan; report more free_slots or resolve blockers"})
        return {
            "session_id": ses.id, "plan_id": ses.plan_id,
            "based_on_version": ses.plan_version, "status": status,
            "decision": decision, "source": source, "reason": reason,
            "retrieval": ses.retrieval,
            "steps": (extra_steps or []) + ses.contract[ses.cursor:],
            "remaining_moves": len(ses.moves),
            "info_requests": requests, "deferred": deferred,
            "state_source": ses.state_source,
        }

    def _finish(self, ses: Session, kind: str, resp: dict, t: dict, t_start: float) -> dict:
        t["total_ms"] = (time.perf_counter() - t_start) * 1e3
        resp["timing_ms"] = {k: round(v, 4) for k, v in t.items()}
        rec = {"kind": kind, "decision": resp["decision"], "source": resp["source"],
               "status": resp["status"], "objects": len(ses.scene.objects) if ses.scene else 0,
               **t}
        self.log.append(rec)
        ses.history.append({"kind": kind, "version": ses.plan_version,
                            "decision": resp["decision"], "status": resp["status"]})
        return resp

    @staticmethod
    def _apply_delta(raw: dict, delta: dict) -> dict:
        """Patch the previous scene message. Objects are matched by id."""
        raw = copy.deepcopy(raw)
        objs = {o["id"]: o for o in raw.get("objects", [])}
        for oid in delta.get("removed", []):
            objs.pop(oid, None)
        for o in delta.get("objects", []):           # added or changed objects (partial dicts)
            objs.setdefault(o["id"], {"id": o["id"]}).update(o)
        raw["objects"] = list(objs.values())
        for k in ("free_slots", "zones", "occluded", "holding", "task", "goal"):
            if k in delta:
                raw[k] = delta[k]
        return raw

    # ------------------------------------------------------------ API operations
    def plan(self, sid: str, scene_msg: dict) -> dict:
        """New task or forced full replan from a complete scene message."""
        t_start, t = time.perf_counter(), {}
        ses = self._session(sid)
        with ses.lock:
            for oid in scene_msg.get("verified", []):
                ses.quarantined.discard(oid)
                ses.retries.pop(oid, None)
            scene = self._load(ses, scene_msg)
            t["parse_ms"] = (time.perf_counter() - t_start) * 1e3
            if ses.scene is not None and scene.version < ses.scene.version:
                raise Stale(f"scene_version {scene.version} < current {ses.scene.version}")
            ses.raw, ses.scene, ses.state_source = scene_msg, scene, "observed"
            return self._finish(ses, "plan", self._make_plan(ses, "new", t), t, t_start)

    def scene_update(self, sid: str, msg: dict) -> dict:
        """A new observation: full scene, or {"scene_version": v, "delta": {...}}.

        Keeps the current plan if its remaining steps still validate on the new
        scene and still reach the goal; otherwise replans from the observed state.
        """
        t_start, t = time.perf_counter(), {}
        ses = self._session(sid)
        with ses.lock:
            if ses.scene is None:
                raise SceneError("no scene yet for this session; call plan first")
            version = int(msg.get("scene_version", -1))
            if version <= ses.scene.version:
                raise Stale(f"scene_version {version} <= current {ses.scene.version}")
            raw = self._apply_delta(ses.raw, msg["delta"]) if "delta" in msg else dict(msg)
            raw["scene_version"] = version
            for k in ("task", "goal"):               # the task belongs to the session, not to one
                if k not in raw and k in ses.raw:    # observation: keep it until a new one is sent
                    raw[k] = ses.raw[k]
            for oid in msg.get("verified", []):
                ses.quarantined.discard(oid)
                ses.retries.pop(oid, None)
            old_deferred = ses.scene.deferred()
            scene = self._load(ses, raw)
            t["parse_ms"] = (time.perf_counter() - t_start) * 1e3
            ses.raw, ses.scene, ses.state_source = raw, scene, "observed"
            tv = time.perf_counter()
            keep = False
            if ses.moves and not ses.awaiting_observation:
                v = validate(scene, ses.moves)
                keep = v.ok and scene.deferred() == old_deferred
            t["check_ms"] = (time.perf_counter() - tv) * 1e3
            if keep:
                # Same moves, but re-issue the contract steps from the observed state.
                ses.contract = to_contract(scene, ses.moves)
                ses.cursor = 0
                ses.plan_version = version
                ses.plan_id = uuid.uuid4().hex[:12]   # step numbers restart, so the id changes
                resp = self._response(ses, "ok", "kept", "kept")
            else:
                ses.replans += 1
                resp = self._make_plan(ses, "replanned", t)
            return self._finish(ses, "scene_update", resp, t, t_start)

    def feedback(self, sid: str, msg: dict) -> dict:
        """Outcome of one contract step.

        msg: {"plan_id", "step", "status": "success"|"failure",
              "failure": "grasp_failed"|"place_failed"|"dropped"|"unreachable"|"collision"|...,
              "scene": optional full observation taken after the action}
        """
        t_start, t = time.perf_counter(), {}
        ses = self._session(sid)
        with ses.lock:
            if msg.get("plan_id") != ses.plan_id:
                raise Stale(f"feedback for plan {msg.get('plan_id')}, current is {ses.plan_id}")
            step_no = int(msg["step"])
            if ses.cursor >= len(ses.contract) or ses.contract[ses.cursor]["step"] != step_no:
                raise Stale(f"feedback for step {step_no}, expected "
                            f"{ses.contract[ses.cursor]['step'] if ses.cursor < len(ses.contract) else None}")
            step = ses.contract[ses.cursor]
            obj = step["object"]
            other = step.get("from") if step["action"] == "PICK" else step.get("destination")
            ctx = self.stats.context(ses.scene, "PICK" if step["action"] == "PICK" else "PLACE",
                                     obj, other)
            ok = msg["status"] == "success"
            self.stats.record(ctx, ok)

            if ok:
                # Predicted effect; an attached observation below overrides it.
                if step["action"] == "PICK":
                    ses.scene.on[obj] = HELD
                else:
                    apply(ses.scene, ses.moves[0])
                    ses.moves = ses.moves[1:]
                ses.cursor += 1
                ses.state_source = "predicted"
                ses.retries.pop(obj, None)
                if msg.get("scene"):
                    resp = self._observe_after_step(ses, msg["scene"], t)
                elif not ses.moves and ses.scene.goal_reached():
                    if ses.case:
                        ses.case.exec_ok += 1
                    status = "partial" if ses.scene.deferred() else "done"
                    resp = self._response(ses, status, "continue", "kept")
                else:
                    resp = self._response(ses, "ok", "continue", "kept")
                return self._finish(ses, "feedback", resp, t, t_start)

            # ---------------- failure handling
            if ses.case:
                ses.case.exec_fail += 1
            failure = msg.get("failure", "unknown")
            if msg.get("scene"):                      # an observation settles what happened
                ses.retries[obj] = ses.retries.get(obj, 0) + 1
                if ses.retries[obj] > self.max_retries or failure == "unreachable":
                    ses.quarantined.add(obj)
                resp = self._observe_after_step(ses, msg["scene"], t, force_replan=True)
                resp["decision"] = "replanned_after_failure"
                return self._finish(ses, "feedback", resp, t, t_start)
            if failure in ("grasp_failed", "place_failed"):
                ses.retries[obj] = ses.retries.get(obj, 0) + 1
                if ses.retries[obj] <= self.max_retries and self.stats.p_success(ctx) >= self.min_retry_p:
                    retry = {"step": step["step"], "action": "RETRY_ACTION", "retry_of": step}
                    resp = self._response(ses, "ok", "retry", "kept",
                                          f"{failure}; attempt {ses.retries[obj] + 1}")
                    resp["steps"] = [retry] + ses.contract[ses.cursor + 1:]
                    return self._finish(ses, "feedback", resp, t, t_start)
                ses.quarantined.add(obj)              # stop insisting on this object
                ses.scene = self._requarantine(ses)
                ses.replans += 1
                resp = self._make_plan(ses, "replanned_after_failure", t)
                resp["reason"] = f"{obj}: {failure} after {ses.retries[obj]} attempts; planning around it"
                return self._finish(ses, "feedback", resp, t, t_start)
            if failure == "unreachable":
                ses.quarantined.add(obj)
                ses.scene = self._requarantine(ses)
                ses.replans += 1
                resp = self._make_plan(ses, "replanned_after_failure", t)
                return self._finish(ses, "feedback", resp, t, t_start)
            # dropped / collision / unknown: the symbolic state can no longer be trusted
            ses.awaiting_observation = True
            ses.moves, ses.contract, ses.cursor = [], [], 0
            resp = self._response(ses, "needs_observation", "verify", "",
                                  f"{failure}: state unknown until the scene is observed again",
                                  extra_steps=[{"step": 0, "action": "VERIFY_SCENE"}])
            return self._finish(ses, "feedback", resp, t, t_start)

    def _requarantine(self, ses: Session) -> Scene:
        """Rebuild the scene with the current quarantine set (objects become unreachable)."""
        from dataclasses import replace
        s = ses.scene.copy()
        s.objects = {k: (replace(o, reachable=False) if k in ses.quarantined else o)
                     for k, o in s.objects.items()}
        return s

    def _observe_after_step(self, ses: Session, raw: dict, t: dict, force_replan: bool = False) -> dict:
        """Compare an observation taken after a step with the predicted state."""
        version = int(raw.get("scene_version", -1))
        if version <= ses.scene.version:
            raise Stale(f"scene_version {version} <= current {ses.scene.version}")
        raw = dict(raw)
        for k in ("task", "goal"):
            if k not in raw and k in ses.raw:
                raw[k] = ses.raw[k]
        predicted = ses.scene.on
        scene = self._load(ses, raw)
        matches = scene.on == predicted and set(scene.objects) == set(ses.scene.objects) \
            and scene.deferred() == ses.scene.deferred() and not force_replan
        ses.raw, ses.scene, ses.state_source = raw, scene, "observed"
        if matches and (not ses.moves or validate(scene, ses.moves).ok):
            ses.plan_version = version
            if not ses.moves and scene.goal_reached():
                if ses.case:
                    ses.case.exec_ok += 1
                return self._response(ses, "partial" if scene.deferred() else "done",
                                      "continue", "kept")
            return self._response(ses, "ok", "continue", "kept")
        ses.replans += 1
        return self._make_plan(ses, "replanned", t)

    # ------------------------------------------------------------ introspection
    def state(self, sid: str) -> dict:
        ses = self._session(sid)
        return {"session_id": sid, "scene_version": ses.scene.version if ses.scene else None,
                "plan_id": ses.plan_id, "plan_version": ses.plan_version,
                "remaining_moves": [str(m) for m in ses.moves], "replans": ses.replans,
                "quarantined": sorted(ses.quarantined), "state_source": ses.state_source,
                "awaiting_observation": ses.awaiting_observation, "history": ses.history[-50:]}

    def metrics(self) -> dict:
        def pct(xs, p):
            xs = sorted(xs)
            return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None
        out = {"requests": len(self.log), "cases": len(self.cases),
               "case_stats": dict(self.cases.stats), "action_stats": self.stats.dump()}
        for kind in ("plan", "scene_update", "feedback"):
            xs = [r["total_ms"] for r in self.log if r["kind"] == kind]
            if xs:
                out[kind] = {"n": len(xs), "p50_ms": pct(xs, .5), "p95_ms": pct(xs, .95),
                             "p99_ms": pct(xs, .99)}
        return out
