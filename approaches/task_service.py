"""The server's logic in one place: understand the task, plan, validate, replan.

    TaskService.task(session, {"scene": ..., "instruction": "..."} )   words in
    TaskService.task(session, {"scene": ..., "task": {...}} )          structured task in
    TaskService.task(session, {"scene": ...} )                         default: sort by colour
    TaskService.task(session, {"images_b64": [...], "instruction": ...})  the model reads the scene

The scene may come in any form scene_understanding.py reads (planner JSON, open-world /
VLM JSON, the simulator export, plain detections with positions). It is turned into the
planner's scene first; the response says what was inferred ("scene_understanding").
    TaskService.scene_update(session, observation)                     keep or replan
    TaskService.feedback(session, step outcome)                        continue, retry, replan

Modes (for comparing approaches with the same robot-side contract):
    hybrid     instruction -> task (rules, then model) -> symbolic planner     [default]
    vlm_plan   instruction -> task as above, then the MODEL writes the plan;
               the validator checks it and sends errors back (up to k rounds)
    symbolic   no model at all: rule parser and structured tasks only

In every mode the plan is replayed by the validator against the current scene
and the task's constraints before it is returned, and the returned steps are
checked against the knowledge base of robot operations.
"""
from __future__ import annotations

import time
from pathlib import Path

from .llm import LLM
from .scene_understanding import SceneReading, match_ids, perceive, rename_ids, task_ids, understand_scene
from .understand import MEMORY_PATH, ExampleMemory, TaskInterpreter, Understanding, load_kb
from semplan.scene import SceneError, parse_scene
from semplan.service import PlanningService, Stale  # noqa: F401  (re-exported for the server)

MODES = ("hybrid", "vlm_plan", "symbolic")


def check_contract(steps: list[dict], kb: dict) -> str | None:
    """Every returned step must be a known operation with its parameters."""
    ops = {o["name"]: o for o in kb["operations"]}
    for s in steps:
        op = ops.get(s.get("action"))
        if op is None:
            return f"step {s.get('step')}: {s.get('action')!r} is not an operation of the robot"
        missing = [p for p in op["params"] if p not in s and p != "target"]
        if missing:
            return f"step {s.get('step')}: {op['name']} lacks {', '.join(missing)}"
    return None


def from_openworld(sc: dict) -> dict:
    """Convert the scene JSON of openworld.py (image perception path) to a semplan scene message."""
    colours = ("red", "blue", "green", "yellow", "orange", "purple", "black", "white", "pink", "brown")
    shapes = {"cube": "cube", "block": "cube", "box": "cube", "cylinder": "cylinder", "prism": "prism"}
    support = sc.get("support", {})
    objects = []
    for o in sc.get("objects", []):
        label = (o.get("label") or o["id"]).lower().replace("_", " ")
        e = {"id": o["id"], "color": next((c for c in colours if c in label), None),
             "type": next((t for w, t in shapes.items() if w in label), "cube")}
        on = support.get(o["id"], "table")
        if on == "unknown" or o["id"] in sc.get("hidden", []):
            e["pose_confidence"] = 0.0
        else:
            e["on"] = on
        if e["color"] is None:
            e["confidence"] = 0.0
        objects.append(e)
    zones = []
    for p in sc.get("places", []):
        label = (p.get("label") or p["id"]).lower()
        zones.append({"id": p["id"], "color": next((c for c in colours if c in label), p["id"]),
                      "capacity": int(p.get("capacity", 3))})
    return {"objects": objects, "zones": zones, "free_slots": list(sc.get("free_slots", []))}


class TaskService:
    def __init__(self, llm: LLM | None = None, mode: str = "hybrid", order: str | None = None,
                 memory_path: Path | None = MEMORY_PATH, plan_rounds: int = 3, use_cache: bool = True,
                 text_llm: LLM | None = None):
        # llm: the vision-language model (images, and text when no text_llm is given).
        # text_llm: a separate text model for instructions, e.g. OllamaLLM("llama3.1").
        text_llm = text_llm or llm
        # With a model, the model understands the sentence and the rule parser is only a
        # fallback for when the model fails; without a model only the rules are available.
        order = order or ("llm,rules" if text_llm is not None else "rules")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "vlm_plan" and llm is None:
            raise ValueError("mode vlm_plan needs a model")
        self.mode, self.llm = mode, llm
        self.kb = load_kb()
        self.memory = ExampleMemory(memory_path)
        self.text_llm = text_llm
        self.interpreter = TaskInterpreter(None if mode == "symbolic" else text_llm, self.memory,
                                           "rules" if mode == "symbolic" or text_llm is None else order,
                                           use_cache=use_cache)
        strategy = None
        if mode == "vlm_plan":
            from .llm_online import LLMModuloStrategy
            strategy = LLMModuloStrategy(llm, plan_rounds)
        self.planning = PlanningService(strategy=strategy, use_cases=mode != "vlm_plan")
        self.sessions: dict[str, dict] = {}            # sid -> instruction, task, source
        self.log: list[dict] = []
        self.pending: dict[str, dict] = {}             # sid -> task request waiting for a new view

    # ------------------------------------------------------------ new task
    def task(self, sid: str, msg: dict) -> dict:
        t0 = time.perf_counter()
        reading, perception = self._read_scene(msg)
        scene_msg = dict(reading.scene)
        instruction = msg.get("instruction")
        if msg.get("task") is not None:
            scene_msg["task"] = msg["task"]
            u = Understanding(msg["task"], "structured")
        elif instruction:
            bare = {k: v for k, v in scene_msg.items() if k not in ("task", "goal")}
            parsed = parse_scene(bare)
            u = self.interpreter.understand(instruction, parsed)
            if u.task is None:
                # The words may name an object whose colour the camera could not see yet:
                # ask for another view of the unidentified objects and try again with it.
                unseen = sorted(o for o, ob in parsed.objects.items() if not parsed.trusted(o))
                waiting = bool(unseen) and u.question.startswith("there is no")
                status = "needs_observation" if waiting else "clarify"
                self.log.append({"kind": "task", "status": status, "source": u.source,
                                 "understand_ms": u.ms, "llm_calls": u.llm_calls})
                if waiting:
                    self.pending[sid] = {k: v for k, v in msg.items() if k not in ("scene", "images", "images_b64")}
                    self.pending[sid]["version"] = parsed.version
                return {"session_id": sid, "status": status, "decision": "none", "steps": [],
                        "question": u.question, "task": None,
                        "info_requests": [{"action": "REQUEST_NEW_VIEW", "target": o,
                                           "reason": "the instruction may refer to this unidentified object"}
                                          for o in unseen] if waiting else [],
                        "understanding": self._describe(u, instruction),
                        "scene_understanding": reading.report(),
                        "timing_ms": {"understand_ms": round(u.ms, 4),
                                      "total_ms": round((time.perf_counter() - t0) * 1e3, 4)}}
            self.pending.pop(sid, None)
            scene_msg["task"] = u.task
            scene_msg.pop("goal", None)
        else:
            u = Understanding(scene_msg.get("task"), "structured" if scene_msg.get("task") else "default")
        calls_before = self.llm.calls if self.llm is not None else 0
        resp = self.planning.plan(sid, scene_msg)
        plan_calls = (self.llm.calls - calls_before) if self.llm is not None else 0
        self.sessions[sid] = {"instruction": instruction, "task": scene_msg.get("task"), "source": u.source,
                              "remembered": False}
        resp = self._finish(sid, resp, u, instruction, t0, plan_calls)
        resp["scene_understanding"] = reading.report()
        if perception:
            resp["scene_understanding"]["perception"] = perception
        return resp

    # ------------------------------------------------------------ reading the scene
    def _read_scene(self, msg: dict) -> tuple[SceneReading, dict | None]:
        images = msg.get("images_b64") or msg.get("images")
        if images and msg.get("scene") is None:
            if self.llm is None or not hasattr(self.llm, "model"):
                raise SceneError("reading a scene from images needs a vision-language model (--model)")
            import base64
            import tempfile
            with tempfile.TemporaryDirectory() as d:
                paths = []
                for i, im in enumerate(images):
                    if msg.get("images_b64"):
                        q = Path(d) / f"view{i}.jpg"
                        q.write_bytes(base64.b64decode(im))
                        paths.append(q)
                    else:
                        paths.append(Path(im))
                t = time.perf_counter()
                reading, info = perceive(paths, self.llm.model)
            return reading, {"model_s": round(time.perf_counter() - t, 3), "model_scene": info["raw"]}
        if msg.get("scene") is None:
            raise SceneError("the request needs a 'scene' or 'images_b64'")
        return understand_scene(msg["scene"]), None

    def _observation(self, sid: str, raw: dict) -> tuple[dict, dict]:
        """A full observation in any supported form -> planner scene with the next version."""
        reading = understand_scene(raw)
        scene = dict(reading.scene)
        given = raw.get("scene_version", raw.get("version"))
        ses = self.planning.sessions.get(sid)
        if not isinstance(given, (int, float)) and ses is not None and ses.scene is not None:
            scene["scene_version"] = ses.scene.version + 1     # the laptop did not number it
        for k in ("verified",):
            if k in raw:
                scene[k] = raw[k]
        report = reading.report()
        self._carry_ids(sid, scene, report)
        return scene, report

    def _carry_ids(self, sid: str, scene: dict, report: dict) -> None:
        """The robot side may number objects differently in every observation. Match the
        objects the task names to the new ids and rename them in the task, so the answer
        always uses the robot's newest ids. Unmatched -> report["unmatched"] (ask to look)."""
        ses = self.planning.sessions.get(sid)
        if ses is None or not ses.raw:
            return
        old = ses.raw.get("objects", [])
        old_ids = {o["id"] for o in old}
        new_ids = {o["id"] for o in scene["objects"]}
        if old_ids <= new_ids:
            return
        mapping = match_ids(old, scene["objects"])
        named = task_ids(ses.raw.get("task") or {}) & old_ids
        missing = sorted(o for o in named if o not in mapping)
        renamed = {a: b for a, b in mapping.items() if a != b}
        if renamed:
            if ses.raw.get("task"):
                task = rename_ids(ses.raw["task"], renamed)
                ses.raw = {**ses.raw, "task": task}
                if sid in self.sessions:
                    self.sessions[sid]["task"] = task
            ses.quarantined = {renamed.get(o, o) for o in ses.quarantined}
            ses.retries = {renamed.get(o, o): n for o, n in ses.retries.items()}
            report["renamed"] = renamed
        if missing:
            report["unmatched"] = missing

    def _cannot_match(self, sid: str, report: dict) -> dict:
        ses = self.planning.sessions[sid]
        return {"session_id": sid, "plan_id": ses.plan_id, "status": "needs_observation",
                "decision": "none", "steps": [],
                "reason": "the task names objects that are not in the new observation: "
                          + ", ".join(report["unmatched"]),
                "info_requests": [{"action": "VERIFY_SCENE", "target": o, "reason": "object of the task not found"}
                                  for o in report["unmatched"]],
                "task": self.sessions.get(sid, {}).get("task"), "scene_understanding": report}

    # ------------------------------------------------------------ during execution
    def scene_update(self, sid: str, msg: dict) -> dict:
        if sid in self.pending and "delta" not in msg:           # the task waited for this view
            req = dict(self.pending.pop(sid))
            req.pop("version", None)
            return self.task(sid, {**req, "scene": msg})
        if "delta" in msg:                                     # deltas use the planner's ids
            return self._after(sid, self.planning.scene_update(sid, msg))
        scene, report = self._observation(sid, msg)
        if report.get("unmatched"):
            return self._cannot_match(sid, report)
        resp = self._after(sid, self.planning.scene_update(sid, scene))
        resp["scene_understanding"] = report
        return resp

    def feedback(self, sid: str, msg: dict) -> dict:
        report = None
        if msg.get("scene") is not None:
            scene, report = self._observation(sid, msg["scene"])
            if report.get("unmatched"):
                return self._cannot_match(sid, report)
            msg = {**msg, "scene": scene}
        resp = self._after(sid, self.planning.feedback(sid, msg))
        if report:
            resp["scene_understanding"] = report
        return resp

    def _after(self, sid: str, resp: dict) -> dict:
        ses = self.sessions.get(sid, {})
        resp["task"] = ses.get("task")
        problem = check_contract(resp["steps"], self.kb)
        if problem:
            resp.update(status="rejected", reason=problem, steps=[])
        # A task that was understood from words and then carried out becomes an example
        # for future instructions (retrieval-augmented understanding).
        if resp["status"] == "done" and ses.get("instruction") and ses.get("source") in ("llm", "rules") \
                and not ses.get("remembered"):
            self.memory.add(ses["instruction"], self.planning.sessions[sid].scene, ses["task"])
            ses["remembered"] = True
        return resp

    def _describe(self, u: Understanding, instruction: str | None) -> dict:
        return {"instruction": instruction, "source": u.source, "llm_calls": u.llm_calls,
                "rounds": u.rounds, "examples_used": u.examples_used, "errors": u.errors,
                "ms": round(u.ms, 4)}

    def _finish(self, sid: str, resp: dict, u: Understanding, instruction, t0: float, plan_calls: int) -> dict:
        resp = self._after(sid, resp)
        scene = self.planning.sessions[sid].scene
        resp["understanding"] = self._describe(u, instruction)
        resp["task_description"] = scene.task.describe() if scene.task is not None else \
            "every object must end in the zone of its own colour"
        resp["mode"] = self.mode
        resp["llm_calls"] = {"understand": u.llm_calls, "plan": plan_calls}
        resp["timing_ms"]["understand_ms"] = round(u.ms, 4)
        resp["timing_ms"]["total_ms"] = round((time.perf_counter() - t0) * 1e3, 4)
        self.log.append({"kind": "task", "status": resp["status"], "source": u.source, "planner": resp["source"],
                         "understand_ms": u.ms, "total_ms": resp["timing_ms"]["total_ms"],
                         "llm_calls": u.llm_calls + plan_calls})
        return resp

    def metrics(self) -> dict:
        m = self.planning.metrics()
        m["mode"] = self.mode
        m["understanding"] = dict(self.interpreter.stats)
        m["examples_in_memory"] = len(self.memory.items)
        if self.llm is not None:
            m["llm_calls_total"] = self.llm.calls
            if hasattr(self.llm, "generate_s"):
                m["llm_generate_s_total"] = round(self.llm.generate_s, 3)
        return m
