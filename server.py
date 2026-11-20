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
  POST /replan          {"request_id", "scene"}: the scene changed -> keep the rest of the plan or replan

Camera (scripts/camera_client.py streams a local camera to the server):
  POST /stream/{cam}         one JPEG frame (body = image/jpeg); the server keeps the latest
  GET  /stream/{cam}.mjpg    live view in a browser;  GET /stream/{cam}/latest.jpg  last frame
  GET  /streams              cameras that are streaming, frame age, rate
  POST /camera/plan          {"camera_id", "instruction"}: latest frame -> VLM -> scene -> plan
  POST /camera/replan        {"camera_id", "request_id"}: new frame -> VLM -> scene -> keep or replan
  GET  /camera               a page with the live view, a task box and the plan

Image detail (every request with images; see image_resolution.py):
  "resolution": "low" | "medium" | "high" | "full" | <max side in px> | "auto"
  "crop": [x0, y0, x1, y1]       fractions 0..1, keep only the work area
  "auto" tries low, then medium, then high, and stops at the first reliable scene.
  The response has a "resolution" field with what was used and every level tried.

Plan quality (every ok plan; see plan_quality.py): "quality" says whether the plan is
proven shortest, the lower bound on moves, and which objects are moved more than once.

Experience database (experience_db.py, on by default, --no-memory to switch off):
  earlier answers are reused when the same model gets the same input: perception (same image
  bytes), goal (same instruction about the same objects), plan (same planning problem).
  GET /memory  what it holds;  POST /memory/clear[?kind=goal]  empty it.
  The response has a "memory" field: hit or miss per stage.

Graphs: GET /dashboard (or python run_report.py) shows where the time goes per request.

Every request is logged in logs/requests/<id>/ (request.json, images, response.json).
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
import yaml
from pydantic import BaseModel, Field

import image_resolution
import openworld

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / 'logs' / 'requests'


class PlanRequest(BaseModel):
    instruction: str | None = Field(None, examples=['put the yellow block in the box'])
    scene: dict | None = Field(None, description='scene abstraction (see openworld.py)')
    images_b64: list[str] | None = Field(None, description='fallback when no scene is sent')
    goal: list | None = Field(None, description='optional: skip the goal model call')
    constraints: dict | None = Field(None, description='optional: no_table, keep, never_on, first '
                                                       '(normally read from the instruction by the model)')
    planner: str = 'auto'
    attempts: int = Field(3, ge=1, le=5, description='retries when the model answer is unusable')
    resolution: str | int | None = Field(None, description='image detail: low | medium | high | full | '
                                         '<max side in px> | auto (low first, more only if needed)')
    crop: list[float] | None = Field(None, description='[x0, y0, x1, y1] fractions 0..1: the work area')
    places: list | None = Field(None, description='fixed places of the lab (zones, boxes) to add to what the camera '
                                              'sees, e.g. [{"id": "red_zone", "color": "red", "capacity": 4}]')
    ordered: bool | None = Field(None, description='the goal list is a sequence (first A, then B ...): goals are met '
                                 'in this order and an earlier goal is never undone. Default: as the model read it')
    compact: bool = Field(False, description='answer with only request_id, status, plan ("action arg ..." strings) '
                          'and quality (moves, optimal); the full answer is still logged for /replan')


class ReplanRequest(BaseModel):
    request_id: str = Field(..., description='request_id of the /plan (or previous /replan) answer')
    scene: dict = Field(..., description='the scene as it is NOW')
    planner: str = 'auto'
    failed: list | None = Field(None, description='steps the robot could not do, e.g. ["(pick-from-item a b)"], '
                                '[{"action": ..., "args": [...]}] or [{"step": 3}]; they are never proposed again')
    compact: bool = Field(False, description='answer with only request_id, status, plan ("action arg ..." strings) '
                          'and quality (moves, optimal); the full answer is still logged for /replan')


class CameraPlanRequest(BaseModel):
    camera_id: str = 'cam0'
    instruction: str
    max_age_s: float = Field(5.0, description='refuse frames older than this')
    planner: str = 'auto'
    attempts: int = Field(2, ge=1, le=5)
    resolution: str | int | None = Field(None, description='image detail: low | medium | high | full | '
                                         '<max side in px> | auto (low first, more only if needed)')
    crop: list[float] | None = Field(None, description='[x0, y0, x1, y1] fractions 0..1: the work area')


class CameraReplanRequest(BaseModel):
    camera_id: str = 'cam0'
    request_id: str = Field(..., description='request_id of the plan to update')
    max_age_s: float = 5.0
    planner: str = 'auto'
    resolution: str | int | None = Field(None, description='image detail: low | medium | high | full | '
                                         '<max side in px> | auto (low first, more only if needed)')
    crop: list[float] | None = Field(None, description='[x0, y0, x1, y1] fractions 0..1: the work area')


class SolveRequest(BaseModel):
    scene: dict
    goal: list = Field(..., examples=[[{'object': 'yellow_block', 'in': 'box_1'}]])
    planner: str = 'auto'
    constraints: dict | None = Field(None, description='no_table, keep, never_on, first, max_stack_height')
    ordered: bool | None = Field(None, description='the goal list is a sequence (first A, then B ...): goals are met '
                                 'in this order and an earlier goal is never undone. Default: as the model read it')
    compact: bool = Field(False, description='answer with only request_id, status, plan ("action arg ..." strings) '
                          'and quality (moves, optimal); the full answer is still logged for /replan')


def compact_answer(res: dict) -> dict:
    """Only what the robot needs: plan steps as "action arg ..." strings and the quality verdict.
    Questions and refusals keep their message; replans keep their decision."""
    out = {'request_id': res.get('request_id'), 'status': res.get('status')}
    if res.get('status') != 'ok' and res.get('message'):
        out['message'] = res['message']
    for k in ('need', 'decision'):
        if res.get(k):
            out[k] = res[k]
    if res.get('plan') is not None:
        out['plan'] = [' '.join([st['action'], *st['args']]) for st in res['plan']]
    q = res.get('quality') or {}
    if q:
        out['quality'] = {'moves': q.get('moves'), 'optimal': q.get('optimal')}
    if (res.get('timings') or {}).get('total_s') is not None:
        out['total_s'] = res['timings']['total_s']       # server-side pipeline time; header X-Server-Ms = whole request
    return out


class GzipRequests:
    """Accept request bodies sent with Content-Encoding: gzip (the robot gzips by default)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or dict(scope.get('headers') or []).get(b'content-encoding', b'').lower() != b'gzip':
            return await self.app(scope, receive, send)
        body, more = b'', True
        while more:
            msg = await receive()
            body += msg.get('body', b'')
            more = msg.get('more_body', False)
        try:
            data = gzip.decompress(body)
        except (OSError, EOFError):
            return await PlainTextResponse('request body says gzip but is not valid gzip', 400)(scope, receive, send)
        headers = [(k, v) for k, v in scope['headers'] if k not in (b'content-encoding', b'content-length')]
        scope = {**scope, 'headers': headers + [(b'content-length', str(len(data)).encode())]}
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {'type': 'http.request', 'body': data, 'more_body': False}
            return await receive()
        return await self.app(scope, replay, send)


class GzipAnswers:
    """gzip answers for clients that send Accept-Encoding: gzip; images and camera streams stay as they are."""

    def __init__(self, app):
        from starlette.middleware.gzip import GZipMiddleware
        self.app, self.gz = app, GZipMiddleware(app, minimum_size=500)

    async def __call__(self, scope, receive, send):
        path = scope.get('path', '') if scope['type'] == 'http' else ''
        if path and not path.startswith(('/stream', '/images', '/dashboard/files')):
            return await self.gz(scope, receive, send)
        return await self.app(scope, receive, send)


def create_app(vlm, max_visual_tokens: int = 1024, log_dir: Path = LOG_DIR, memory=None) -> FastAPI:
    app = FastAPI(title='VLM + PDDL planning server', version='1.0')
    # allow the web page also when it is opened as a file or from another host/port
    app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'],
                       expose_headers=['X-Server-Ms'])
    app.add_middleware(GzipAnswers)
    app.add_middleware(GzipRequests)

    @app.middleware('http')
    async def server_time(request: Request, call_next):
        """X-Server-Ms: milliseconds from the request arriving to the answer leaving the handler, so a
        client can split its own round trip into server time and network time."""
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers['X-Server-Ms'] = f'{(time.perf_counter() - t0) * 1000:.1f}'
        return response
    gpu_lock = threading.Lock()           # one GPU: one model call at a time
    state = {'started': time.time(), 'requests': 0}

    class LockedVLM:                      # wraps every generate() call in the lock
        def generate(self, *a, **kw):
            with gpu_lock:
                return vlm.generate(*a, **kw)

    model = LockedVLM() if vlm is not None else None

    def plan_images(images, resolution, crop, **kw):
        """Run the pipeline; with images, at the requested resolution (see image_resolution.py)."""
        if not images or kw.get('scene') is not None:
            return openworld.run_until_ok(images=images, vlm=model, max_visual_tokens=max_visual_tokens,
                                          memory=memory, **kw)
        try:
            return image_resolution.run_with_resolution(openworld.run_until_ok, images, resolution, crop,
                                                        max_visual_tokens, vlm=model, memory=memory, **kw)
        except ValueError as e:
            raise HTTPException(422, str(e))

    def perceive_images(images, resolution, crop):
        try:
            return image_resolution.perceive_with_resolution(
                lambda imgs, tokens: openworld.perceive(imgs, model, tokens),
                images, resolution, crop, max_visual_tokens)
        except ValueError as e:
            raise HTTPException(422, str(e))

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

    def finish(rid: str, d: Path, request: dict, result: dict, received_bytes: int, compact: bool = False) -> dict:
        result = {'request_id': rid, **result, 'received_bytes': received_bytes,
                  'model': getattr(vlm, 'key', None)}
        (d / 'request.json').write_text(json.dumps(request, indent=2))
        (d / 'response.json').write_text(json.dumps(result, indent=2))       # always the full answer
        return compact_answer(result) if compact else result

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
        res = plan_images(images, req.resolution, req.crop, attempts=req.attempts,
                          instruction=req.instruction, scene=req.scene, goal=req.goal,
                          planner=req.planner, constraints=req.constraints, places=req.places,
                          ordered=req.ordered)
        return finish(rid, d, {**body, 'images': [p.name for p in images]}, res, received, req.compact)

    @app.post('/plan/upload')
    async def plan_upload(instruction: str = Form(...), files: list[UploadFile] = File(default=[]),
                          scene: str | None = Form(None), planner: str = Form('auto'),
                          attempts: int = Form(3), resolution: str | None = Form(None),
                          crop: str | None = Form(None, description='JSON list [x0, y0, x1, y1]'),
                          compact: bool = Form(False)):
        req = {'instruction': instruction, 'planner': planner, 'resolution': resolution, 'crop': crop}
        try:
            crop_box = json.loads(crop) if crop else None
        except json.JSONDecodeError:
            raise HTTPException(400, 'crop must be a JSON list like [0.1, 0.2, 0.9, 1.0]')
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
        res = plan_images(images, resolution, crop_box, attempts=min(max(attempts, 1), 5),
                          instruction=instruction, scene=req.get('scene'), planner=planner)
        return finish(rid, d, {**req, 'images': [p.name for p in images]}, res, received, compact)

    @app.post('/perceive')
    async def perceive(files: list[UploadFile] = File(...), resolution: str | None = Form(None),
                       crop: str | None = Form(None)):
        if model is None:
            raise HTTPException(503, 'no model loaded')
        rid, d = new_request()
        blobs = [(f.filename, await f.read()) for f in files]
        images = save_images(d, blobs)
        js, err, g, rinfo = perceive_images(images, resolution, json.loads(crop) if crop else None)
        res = {'status': 'error' if err else 'ok', 'message': err, 'model_answer': g['text'],
               'timings': {'perceive_s': g['generate_s']}, 'resolution': rinfo}
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
        res = openworld.run(scene=req.scene, goal=req.goal, planner=req.planner, memory=memory,
                            constraints=req.constraints, ordered=req.ordered)
        return finish(rid, d, req.model_dump(), res, len(json.dumps(req.scene)), req.compact)

    @app.post('/replan')
    def replan(req: ReplanRequest):
        """The scene changed after a plan was made: keep the rest of the plan or make a new one.
        The goal and constraints of the first request are reused; the model is not called."""
        prev = log_dir / Path(req.request_id).name / 'response.json'
        if not prev.exists():
            raise HTTPException(404, 'unknown request_id')
        rid, d = new_request()
        res = openworld.replan(json.loads(prev.read_text()), req.scene, req.planner, memory=memory,
                               failed=req.failed)
        res['replans'] = req.request_id
        return finish(rid, d, req.model_dump(), res, len(json.dumps(req.scene)), req.compact)

    # ------------------------------------------------------------ camera stream
    frames: dict[str, dict] = {}          # camera id -> latest frame, time, count, rate
    frames_lock = threading.Lock()

    def latest(cam: str, max_age_s: float) -> bytes:
        with frames_lock:
            f = frames.get(cam)
        if f is None:
            raise HTTPException(409, f'no frame from camera {cam!r} yet: start scripts/camera_client.py')
        age = time.time() - f['t']
        if age > max_age_s:
            raise HTTPException(409, f'last frame from {cam!r} is {age:.1f} s old (is the client still streaming?)')
        return f['jpeg']

    @app.post('/stream/{cam}')
    async def stream_frame(cam: str, request: Request):
        data = await request.body()
        if len(data) < 4 or data[:2] != b'\xff\xd8':
            raise HTTPException(400, 'body must be one JPEG image (Content-Type: image/jpeg)')
        now = time.time()
        with frames_lock:
            old = frames.get(cam)
            rate = 0.0 if old is None else 0.8 * old['fps'] + 0.2 / max(now - old['t'], 1e-3)
            frames[cam] = {'jpeg': data, 't': now, 'n': (old['n'] + 1) if old else 1, 'fps': rate}
        return {'ok': True, 'camera_id': cam, 'bytes': len(data), 'server_time': now}

    @app.get('/streams')
    def streams():
        with frames_lock:
            return {c: {'age_s': round(time.time() - f['t'], 2), 'frames': f['n'], 'fps': round(f['fps'], 2),
                        'bytes': len(f['jpeg'])} for c, f in frames.items()}

    @app.get('/stream/{cam}/latest.jpg')
    def stream_latest(cam: str):
        return Response(latest(cam, 1e9), media_type='image/jpeg')

    @app.get('/stream/{cam}.mjpg')
    def stream_mjpeg(cam: str):
        def gen():
            last = None
            for _ in range(36000):                    # stops after about an hour
                with frames_lock:
                    f = frames.get(cam)
                if f is not None and f['t'] != last:
                    last = f['t']
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + f['jpeg'] + b'\r\n')
                time.sleep(0.1)
        return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')

    @app.post('/camera/plan')
    def camera_plan(req: CameraPlanRequest):
        """Plan from what the camera sees right now."""
        jpeg = latest(req.camera_id, req.max_age_s)
        rid, d = new_request()
        images = save_images(d, [(f'{req.camera_id}.jpg', jpeg)])
        res = plan_images(images, req.resolution, req.crop, attempts=req.attempts,
                          instruction=req.instruction, planner=req.planner)
        res['camera_id'] = req.camera_id
        return finish(rid, d, {**req.model_dump(), 'images': [p.name for p in images]}, res, len(jpeg))

    @app.post('/camera/replan')
    def camera_replan(req: CameraReplanRequest):
        """The scene may have changed: look again, then keep the rest of the plan or replan."""
        prev = log_dir / Path(req.request_id).name / 'response.json'
        if not prev.exists():
            raise HTTPException(404, 'unknown request_id')
        jpeg = latest(req.camera_id, req.max_age_s)
        rid, d = new_request()
        images = save_images(d, [(f'{req.camera_id}.jpg', jpeg)])
        t0 = time.perf_counter()
        scene, err, g, rinfo = perceive_images(images, req.resolution, req.crop)
        if err:
            res = {'status': 'error', 'message': f'perception: {err}', 'model_answers': {'perceive': g['text']}}
        else:
            res = openworld.replan(json.loads(prev.read_text()), scene, req.planner, memory=memory)
            res.setdefault('timings', {})['perceive_s'] = g.get('generate_s')
        res['timings']['total_s'] = round(time.perf_counter() - t0, 3)
        res['replans'], res['camera_id'], res['resolution'] = req.request_id, req.camera_id, rinfo
        return finish(rid, d, {**req.model_dump(), 'images': [p.name for p in images]}, res, len(jpeg))

    @app.get('/camera', response_class=HTMLResponse)
    def camera_page(cam: str = 'cam0'):
        return CAMERA_PAGE.replace('{{CAM}}', cam)

    @app.get('/memory')
    def memory_stats():
        """What the experience database holds and how often it answered."""
        return memory.stats() if memory is not None else {'enabled': False}

    @app.post('/memory/clear')
    def memory_clear(kind: str | None = None):
        if memory is None:
            raise HTTPException(404, 'the experience database is switched off')
        if kind is not None and kind not in ('scene', 'goal', 'plan'):
            raise HTTPException(422, 'kind must be scene, goal or plan')
        return {'deleted': memory.clear(kind)}

    report_dir = log_dir.parent / 'report'

    @app.get('/dashboard', response_class=HTMLResponse)
    def dashboard(last: int | None = None):
        """Graphs of the logged requests (rebuilt on every visit; see run_report.py)."""
        try:
            import run_report
        except ImportError as e:
            raise HTTPException(503, f'graphs need matplotlib: {e}')
        page = run_report.build(log_dir, report_dir, last, img_prefix='/dashboard/files/')
        return HTMLResponse(page.read_text())

    @app.get('/dashboard/files/{name}')
    def dashboard_file(name: str):
        path = report_dir / Path(name).name
        if path.suffix != '.png' or not path.exists():
            raise HTTPException(404, 'no such chart')
        return FileResponse(path)

    @app.get('/requests/{rid}')
    def get_request(rid: str):
        d = log_dir / Path(rid).name
        if not (d / 'response.json').exists():
            raise HTTPException(404, 'unknown request id')
        return {'request': json.loads((d / 'request.json').read_text()),
                'response': json.loads((d / 'response.json').read_text())}

    return app


CAMERA_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Camera planner</title>
<style>body{font-family:system-ui,sans-serif;margin:16px;background:#f7f7f8;color:#1f2933}
.row{display:flex;gap:16px;flex-wrap:wrap}img{max-width:640px;width:100%;border:1px solid #ccc;background:#000}
input{width:420px;padding:6px}button{padding:6px 12px}pre{background:#fff;border:1px solid #ddd;padding:8px;
max-width:560px;white-space:pre-wrap}</style></head><body>
<h3>Camera {{CAM}} (streamed by scripts/camera_client.py)</h3>
<div class="row"><div><img src="/stream/{{CAM}}.mjpg"></div><div>
<input id="t" placeholder="task, e.g. put the blue cylinder on the yellow block">
<button onclick="plan()">Plan</button> <button onclick="replan()">Scene changed: replan</button>
<pre id="out">type a task and press Plan</pre></div></div>
<script>let rid=null;
function show(r){rid=r.request_id||rid;const s=r.scene&&r.scene.support?Object.entries(r.scene.support)
.map(([a,b])=>a+' on '+b).join('\\n'):'';const p=(r.plan||[]).map((a,i)=>(i+1)+'. '+a.pddl).join('\\n');
document.getElementById('out').textContent='status: '+r.status+(r.message?' - '+r.message:'')+
(r.decision?'\\ndecision: '+r.decision:'')+'\\n\\nseen:\\n'+s+'\\n\\ngoal: '+JSON.stringify(r.goal||null)+
'\\nconstraints: '+JSON.stringify(r.constraints||null)+'\\n\\nplan:\\n'+(p||'(none)');}
async function post(u,b){document.getElementById('out').textContent='working (model)...';
const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
const j=await r.json();show(r.ok?j:{status:'error',message:j.detail});}
function plan(){post('/camera/plan',{camera_id:'{{CAM}}',instruction:document.getElementById('t').value});}
function replan(){if(!rid)return alert('plan first');post('/camera/replan',{camera_id:'{{CAM}}',request_id:rid});}
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default=os.environ.get('MODEL', 'qwen3-vl-8b'),
                    help='key from models.yaml, or "mock" (no GPU)')
    ap.add_argument('--quant', default=os.environ.get('QUANT', '4bit'), choices=['4bit', '8bit', 'bf16'])
    ap.add_argument('--host', default=os.environ.get('HOST', '0.0.0.0'))
    ap.add_argument('--port', type=int, default=int(os.environ.get('PORT', 8010)))
    ap.add_argument('--max-visual-tokens', type=int, default=int(os.environ.get('MAX_VISUAL_TOKENS', 1024)))
    ap.add_argument('--memory', default=os.environ.get('MEMORY', str(ROOT / 'logs' / 'experience.sqlite')),
                    help='experience database file (scene, goal and plan answers of earlier requests)')
    ap.add_argument('--no-memory', action='store_true',
                    default=os.environ.get('NO_MEMORY', '') not in ('', '0'),
                    help='switch the experience database off (or set NO_MEMORY=1)')
    ap.add_argument('--memory-examples', type=int, default=int(os.environ.get('MEMORY_EXAMPLES', 0)),
                    help='put this many similar earlier instructions into the goal prompt (0 = off)')
    args = ap.parse_args()

    import uvicorn
    if args.model == 'mock':
        vlm = openworld.RuleMock()
    else:
        from vlm import load_model
        print(f'loading {args.model} ({args.quant}) ...', flush=True)
        vlm = load_model(args.model, args.quant)
        print(f'loaded in {vlm.load_s:.1f} s, weights {vlm.weights_vram_gb:.1f} GB VRAM', flush=True)
    memory = None
    if not args.no_memory:
        from experience_db import ExperienceDB
        memory = ExperienceDB(args.memory, model_key=f'{args.model}/{args.quant}', examples=args.memory_examples)
        print(f'experience database: {args.memory} ({memory.stats()["stored"]})', flush=True)
    uvicorn.run(create_app(vlm, args.max_visual_tokens, memory=memory), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
