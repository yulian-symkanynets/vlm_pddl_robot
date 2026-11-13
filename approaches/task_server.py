"""The planning server: instructions or structured tasks in, validated action plans out.

    python -m approaches.task_server                              # no model: rule parser + structured tasks
    python -m approaches.task_server --text-model ollama:llama3.1 # a local text model understands sentences
    python -m approaches.task_server --model qwen3-vl-8b          # GPU machine: images + sentences, models.yaml key
    python -m approaches.task_server --model qwen3-vl-8b --mode vlm_plan     # the model writes the plan

    POST /v1/sessions/{sid}/task       {"scene": {...}, "instruction": "..."}   or  {"scene", "task": {...}}
    POST /v1/sessions/{sid}/scene      new observation (full or delta): plan kept or replanned
    POST /v1/sessions/{sid}/feedback   outcome of one step
    POST /v1/perceive                  {"images_b64": [...]} -> scene message (needs a model)
    GET  /v1/sessions/{sid}            server-side state
    GET  /v1/knowledge                 robot operations, rules, goal and constraint vocabulary
    GET  /v1/metrics                   latency percentiles, how tasks were understood, model calls
    GET  /health

Status codes: 409 = the message refers to an old plan or scene version; 422 = the scene or
task is inconsistent. A task that needs a question back to the user has status "clarify".
"""
from __future__ import annotations

import argparse
import base64
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from .task_service import MODES, TaskService
from semplan.scene import SceneError
from semplan.service import Stale


def create_app(service: TaskService) -> FastAPI:
    app = FastAPI(title="robot task planning server", version="1.0")

    async def call(fn, sid: str, request: Request):
        try:
            return fn(sid, await request.json())
        except Stale as e:
            raise HTTPException(409, str(e))
        except (SceneError, KeyError, TypeError, ValueError) as e:
            raise HTTPException(422, f"{type(e).__name__}: {e}")

    @app.post("/v1/sessions/{sid}/task")
    async def task(sid: str, request: Request):
        return await call(service.task, sid, request)

    @app.post("/v1/sessions/{sid}/scene")
    async def scene(sid: str, request: Request):
        return await call(service.scene_update, sid, request)

    @app.post("/v1/sessions/{sid}/feedback")
    async def feedback(sid: str, request: Request):
        return await call(service.feedback, sid, request)

    @app.get("/v1/sessions/{sid}")
    def state(sid: str):
        return {**service.planning.state(sid), **{k: v for k, v in service.sessions.get(sid, {}).items()
                                                  if k in ("instruction", "task", "source")}}

    @app.get("/v1/knowledge")
    def knowledge():
        return service.kb

    @app.get("/v1/metrics")
    def metrics():
        return service.metrics()

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": service.mode, "model": getattr(service.llm, "key", None)}

    @app.post("/v1/perceive")
    async def perceive(request: Request):
        """Fallback for when the laptop cannot build the scene itself: images -> scene message.
        Uses the perception prompt of openworld.py. Check the result before planning on it."""
        if service.llm is None or not hasattr(service.llm, "model"):
            raise HTTPException(503, "perception needs a vision-language model (--model)")
        body = await request.json()
        import openworld
        from .llm import extract_json
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i, b64 in enumerate(body.get("images_b64", [])):
                p = Path(d) / f"view{i}.jpg"
                p.write_bytes(base64.b64decode(b64))
                paths.append(p)
            if not paths:
                raise HTTPException(422, "send images_b64")
            out = service.llm.model.generate(paths, openworld.perceive_prompt(len(paths)))
        js = extract_json(out["text"])
        if not isinstance(js, dict):
            raise HTTPException(502, "the model did not return a scene")
        from .scene_understanding import understand_scene
        reading = understand_scene(js)
        return {"scene": reading.scene, "scene_understanding": reading.report(), "raw": js,
                "generate_s": out.get("generate_s")}

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", help="model key from models.yaml; omit to run without a model")
    ap.add_argument("--quant", default="4bit")
    ap.add_argument("--text-model", help="separate model for instructions, e.g. ollama:llama3.1 "
                                         "(local Ollama); default: the --model, else rules only")
    ap.add_argument("--mode", default="hybrid", choices=MODES)
    ap.add_argument("--order", default=None,
                    help="task understanding: llm,rules (default with a model) | llm | rules | rules,llm")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8020)
    args = ap.parse_args()
    llm = None
    if args.model:
        from .llm import RepoVLM
        llm = RepoVLM(args.model, args.quant)
        llm.key = args.model
    text_llm = None
    if args.text_model:
        if not args.text_model.startswith("ollama:"):
            raise SystemExit("--text-model must look like ollama:<model>")
        from .llm import OllamaLLM
        text_llm = OllamaLLM(args.text_model.split(":", 1)[1])
    import uvicorn
    uvicorn.run(create_app(TaskService(llm, args.mode, args.order, text_llm=text_llm)),
                host=args.host, port=args.port)


if __name__ == "__main__":
    main()
