"""Planning server: instruction + scene abstraction (or images) -> verified robot plan.

Start (foreground):   python server.py --model qwen3-vl-8b
Start (background):   ./serve.sh start            (see ./serve.sh help)
Without a GPU:        python server.py --model mock
API docs in browser:  http://127.0.0.1:8010/docs

Web page:             http://127.0.0.1:8010/   (upload photos, type the task, see the plan)

Endpoints
  GET  /                the test web page (web/index.html)
  GET  /examples        example cases for the web page
  GET  /health          model, uptime, number of requests
  GET  /domain          the fixed PDDL domain
  POST /plan            JSON  {"instruction", "scene"}            semantic path, no image (main one)
                        JSON  {"instruction", "images_b64": [...]} fallback: server does the perception
  POST /plan/upload     multipart: instruction + files (+ optional scene as JSON text)
  POST /perceive        multipart: files -> scene JSON only (what the laptop encoder will send)
  POST /solve           JSON  {"scene", "goal"}  -> plan, no model at all (test the planner)
  GET  /requests/{id}   a logged request + response

Every request is logged in logs/requests/<id>/ (request.json, images, response.json).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import yaml
from pydantic import BaseModel, Field

import openworld

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / 'logs' / 'requests'


class PlanRequest(BaseModel):
    instruction: str | None = Field(None, examples=['put the yellow block in the box'])
    scene: dict | None = Field(None, description='scene abstraction (see openworld.py)')
    images_b64: list[str] | None = Field(None, description='fallback when no scene is sent')
    goal: list | None = Field(None, description='optional: skip the goal model call')
    planner: str = 'auto'
    attempts: int = Field(3, ge=1, le=5, description='retries when the model answer is unusable')


class SolveRequest(BaseModel):
    scene: dict
    goal: list = Field(..., examples=[[{'object': 'yellow_block', 'in': 'box_1'}]])
    planner: str = 'auto'


def create_app(vlm, max_visual_tokens: int = 1024, log_dir: Path = LOG_DIR) -> FastAPI:
    app = FastAPI(title='VLM + PDDL planning server', version='1.0')
    # allow the web page also when it is opened as a file or from another host/port
    app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
    gpu_lock = threading.Lock()           # one GPU: one model call at a time
    state = {'started': time.time(), 'requests': 0}

    class LockedVLM:                      # wraps every generate() call in the lock
        def generate(self, *a, **kw):
            with gpu_lock:
                return vlm.generate(*a, **kw)

    model = LockedVLM() if vlm is not None else None

    def new_request() -> tuple[str, Path]:
        rid = datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:6]
        d = log_dir / rid
        d.mkdir(parents=True, exist_ok=True)
        state['requests'] += 1
        return rid, d

    def save_images(d: Path, blobs: list[tuple[str, bytes]]) -> list[Path]:
        paths = []
        for k, (name, data) in enumerate(blobs):
            ext = Path(name or '').suffix.lower() or '.jpg'
            p = d / f'image_{k + 1}{ext}'
            p.write_bytes(data)
            paths.append(p)
        return paths

    def finish(rid: str, d: Path, request: dict, result: dict, received_bytes: int) -> dict:
        result = {'request_id': rid, **result, 'received_bytes': received_bytes,
                  'model': getattr(vlm, 'key', None)}
        (d / 'request.json').write_text(json.dumps(request, indent=2))
        (d / 'response.json').write_text(json.dumps(result, indent=2))
        return result

    @app.get('/', include_in_schema=False)
    def index():
        return FileResponse(ROOT / 'web' / 'index.html')

    app.mount('/images', StaticFiles(directory=ROOT / 'images'), name='images')

    @app.get('/examples')
    def examples():
        """Example cases for the web page (scene JSON inline, images as URLs)."""
        base = ROOT / 'examples'
        out = []
        for c in yaml.safe_load((base / 'cases.yaml').read_text())['cases']:
            if c.get('goal'):
                continue                              # planner-only cases have no instruction
            e = {'name': c['name'], 'instruction': c['instruction'], 'expect': c['expect']}
            if c.get('scene'):
                e['scene'] = json.loads((base / c['scene']).read_text())
            if c.get('images'):
                e['images'] = ['/images/' + Path(p).name for p in c['images']]
            out.append(e)
        return out

    @app.get('/health')
    def health():
        return {'status': 'up', 'model': getattr(vlm, 'key', None), 'hf': getattr(vlm, 'hf', None),
                'quant': getattr(vlm, 'quant', None), 'max_visual_tokens': max_visual_tokens,
                'uptime_s': round(time.time() - state['started']), 'requests': state['requests']}

    @app.get('/domain', response_class=PlainTextResponse)
    def domain():
        return openworld.domain_text()

    @app.post('/plan')
    def plan(req: PlanRequest):
        body = req.model_dump(exclude_none=True)
        b64 = body.pop('images_b64', None) if req.scene is None else None
        body.pop('images_b64', None)
        blobs = []
        for k, b in enumerate(b64 or []):
            try:
                blobs.append((f'upload_{k + 1}.jpg', base64.b64decode(b.split(',', 1)[-1], validate=True)))
            except ValueError:
                raise HTTPException(400, f'images_b64[{k}] is not valid base64')
        rid, d = new_request()
        images = save_images(d, blobs)
        received = len(json.dumps(req.scene)) if req.scene else sum(len(x) for _, x in blobs)
        res = openworld.run_until_ok(req.attempts, instruction=req.instruction, scene=req.scene,
                                     images=images, vlm=model, goal=req.goal, planner=req.planner,
                                     max_visual_tokens=max_visual_tokens)
        return finish(rid, d, {**body, 'images': [p.name for p in images]}, res, received)

    @app.post('/plan/upload')
    async def plan_upload(instruction: str = Form(...), files: list[UploadFile] = File(default=[]),
                          scene: str | None = Form(None), planner: str = Form('auto'),
                          attempts: int = Form(3)):
        req = {'instruction': instruction, 'planner': planner}
        received = 0
        if scene:
            try:
                req['scene'] = json.loads(scene)
            except json.JSONDecodeError as e:
                raise HTTPException(400, f'scene is not valid JSON: {e}')
            received += len(scene)
        rid, d = new_request()
        blobs = [(f.filename, await f.read()) for f in files] if not scene else []
        received += sum(len(b) for _, b in blobs)
        images = save_images(d, blobs)
        res = openworld.run_until_ok(min(max(attempts, 1), 5), instruction=instruction,
                                     scene=req.get('scene'), images=images, vlm=model, planner=planner,
                                     max_visual_tokens=max_visual_tokens)
        return finish(rid, d, {**req, 'images': [p.name for p in images]}, res, received)

    @app.post('/perceive')
    async def perceive(files: list[UploadFile] = File(...)):
        if model is None:
            raise HTTPException(503, 'no model loaded')
        rid, d = new_request()
        blobs = [(f.filename, await f.read()) for f in files]
        images = save_images(d, blobs)
        g = model.generate(images, openworld.perceive_prompt(len(images)), max_visual_tokens,
                           context={'mode': 'ow_perceive'})
        js, err = openworld._json_answer(g['text'])
        res = {'status': 'error' if err else 'ok', 'message': err, 'model_answer': g['text'],
               'timings': {'perceive_s': g['generate_s']}}
        if not err:
            try:
                res['scene'] = openworld.scene_to_json(openworld.normalize_scene(js))
                res['scene_bytes'] = len(json.dumps(res['scene']))
            except openworld.SceneError as e:
                res.update(status='error', message=f'scene: {e}', scene_raw=js)
        return finish(rid, d, {'images': [p.name for p in images]}, res,
                      sum(len(b) for _, b in blobs))

    @app.post('/solve')
    def solve(req: SolveRequest):
        rid, d = new_request()
        res = openworld.run(scene=req.scene, goal=req.goal, planner=req.planner)
        return finish(rid, d, req.model_dump(), res, len(json.dumps(req.scene)))

    @app.get('/requests/{rid}')
    def get_request(rid: str):
        d = log_dir / Path(rid).name
        if not (d / 'response.json').exists():
            raise HTTPException(404, 'unknown request id')
        return {'request': json.loads((d / 'request.json').read_text()),
                'response': json.loads((d / 'response.json').read_text())}

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default=os.environ.get('MODEL', 'qwen3-vl-8b'),
                    help='key from models.yaml, or "mock" (no GPU)')
    ap.add_argument('--quant', default=os.environ.get('QUANT', '4bit'), choices=['4bit', '8bit', 'bf16'])
    ap.add_argument('--host', default=os.environ.get('HOST', '0.0.0.0'))
    ap.add_argument('--port', type=int, default=int(os.environ.get('PORT', 8010)))
    ap.add_argument('--max-visual-tokens', type=int, default=int(os.environ.get('MAX_VISUAL_TOKENS', 1024)))
    args = ap.parse_args()

    import uvicorn
    if args.model == 'mock':
        vlm = openworld.RuleMock()
    else:
        from vlm import load_model
        print(f'loading {args.model} ({args.quant}) ...', flush=True)
        vlm = load_model(args.model, args.quant)
        print(f'loaded in {vlm.load_s:.1f} s, weights {vlm.weights_vram_gb:.1f} GB VRAM', flush=True)
    uvicorn.run(create_app(vlm, args.max_visual_tokens), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
