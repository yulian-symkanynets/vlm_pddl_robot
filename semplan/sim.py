"""A small simulated robot + laptop, for closed-loop tests without hardware.

SimRobot owns the ground-truth scene. It executes contract steps with injected
failures, lets a "human" disturb the scene, hides the colour of some objects
until another view is requested, and produces the scene messages the laptop
would send. It replaces the robot side only; the server code is the real one.
"""
from __future__ import annotations

import random

from .scene import HELD, TABLE, Scene, dump_scene, legal_dests
from .service import PlanningService, Stale


class SimRobot:
    def __init__(self, truth: Scene, rng: random.Random, p_pick_fail: float = 0.0,
                 p_drop: float = 0.0, hidden: set[str] | None = None,
                 prism_penalty: float = 0.0):
        self.truth = truth.copy()
        self.rng = rng
        self.p_pick_fail = p_pick_fail
        self.p_drop = p_drop
        self.prism_penalty = prism_penalty        # extra pick failure probability for prisms
        self.hidden = set(hidden or ())           # colour not visible from the wrist camera
        self.version = truth.version
        self.actions = 0
        self.reverified: dict[str, int] = {}      # how often the laptop re-checked an object
        self.max_reverify = 2
        self._verified: list[str] = []

    def observe(self) -> dict:
        self.version += 1
        self.truth.version = self.version
        msg = dump_scene(self.truth)
        for o in msg["objects"]:
            if o["id"] in self.hidden:
                o["color"], o["confidence"] = None, 0.3
        if self._verified:
            msg["verified"], self._verified = self._verified, []
        return msg

    def fulfil(self, request: dict) -> bool:
        """Another camera view reveals a hidden object."""
        if request["action"] in ("REQUEST_NEW_VIEW", "CHANGE_VIEWPOINT") \
                and request.get("target") in self.hidden:
            self.hidden.discard(request["target"])
            return True
        if request["action"] == "VERIFY_SCENE" and request.get("reason") == "repeated_failure":
            o = request["target"]                  # laptop re-measures the pose and clears the object
            if self.reverified.get(o, 0) < self.max_reverify:
                self.reverified[o] = self.reverified.get(o, 0) + 1
                self._verified.append(o)
                return True
        return False

    def execute(self, step: dict) -> dict:
        """Returns {"status", "failure"?}. Mutates the ground truth."""
        if step["action"] == "RETRY_ACTION":
            step = step["retry_of"]
        if step["action"] == "VERIFY_SCENE":
            return {"status": "success"}
        self.actions += 1
        t, obj = self.truth, step["object"]
        if step["action"] == "PICK":
            if obj not in t.objects or obj not in t.clear() or t.held() is not None:
                return {"status": "failure", "failure": "collision"}
            p = self.p_pick_fail + (self.prism_penalty if t.objects[obj].type == "prism" else 0)
            if self.rng.random() < p:
                return {"status": "failure", "failure": "grasp_failed"}
            t.on[obj] = HELD
            return {"status": "success"}
        dst = step["destination"]
        if t.held() != obj:
            return {"status": "failure", "failure": "collision"}
        if self.rng.random() < self.p_drop:
            t.on[obj] = TABLE                       # slipped: lands somewhere on the table
            return {"status": "failure", "failure": "dropped"}
        if dst not in legal_dests(t, obj):
            return {"status": "failure", "failure": "place_failed"}
        t.on[obj] = dst
        return {"status": "success"}

    def human_disturbance(self) -> str | None:
        """A person moves one clear object to another legal place."""
        t = self.truth
        cands = [o for o in sorted(t.clear())]
        if not cands:
            return None
        o = self.rng.choice(cands)
        dests = [d for d in legal_dests(t, o) if d != t.on[o]] + [TABLE]
        t.on[o] = self.rng.choice(dests)
        return o


def run_episode(service: PlanningService, robot: SimRobot, sid: str, rng: random.Random,
                p_disturb: float = 0.0, max_steps: int = 400, verify_each_step: bool = True) -> dict:
    """Closed loop: plan, execute one step at a time, feed back, replan when needed."""
    out = {"goal": False, "steps": 0, "disturbances": 0, "stale": 0, "requests": 0,
           "server_ms": 0.0, "calls": 0, "last_status": ""}

    def account(resp):
        out["server_ms"] += resp["timing_ms"]["total_ms"]
        out["calls"] += 1
        out["last_status"] = resp["status"]
        return resp

    resp = account(service.plan(sid, robot.observe()))
    for _ in range(max_steps):
        fulfilled = [robot.fulfil(r) for r in resp["info_requests"]]
        if any(fulfilled):
            out["requests"] += sum(fulfilled)
            resp = account(service.scene_update(sid, robot.observe()))
            continue
        if not resp["steps"]:
            break
        if rng.random() < p_disturb and robot.human_disturbance():
            out["disturbances"] += 1
            resp = account(service.scene_update(sid, robot.observe()))
            continue
        step = resp["steps"][0]
        result = robot.execute(step)
        out["steps"] += 1
        if step["action"] == "VERIFY_SCENE":
            resp = account(service.scene_update(sid, robot.observe()))
            continue
        msg = {"plan_id": resp["plan_id"], "step": step["step"], **result}
        if verify_each_step and result.get("failure") != "grasp_failed":
            msg["scene"] = robot.observe()
        try:
            resp = account(service.feedback(sid, msg))
        except Stale:
            out["stale"] += 1
            resp = account(service.plan(sid, robot.observe()))
    final = robot.truth
    out["goal"] = final.held() is None and all(
        final.base(o) == z for o in final.objects if (z := final.target(o)) is not None)
    out["replans"] = service.sessions[sid].replans
    out["robot_actions"] = robot.actions
    return out
