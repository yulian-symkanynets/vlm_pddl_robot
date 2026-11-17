"""Stream a local camera to the planning server and get plans from what it sees.

Runs on YOUR computer (the one the camera is plugged into). Needs: pip install opencv-python numpy

    python scripts/camera_client.py --server http://<server-ip>:8010
    python scripts/camera_client.py --server http://<server-ip>:8010 --camera 1      # second camera
    python scripts/camera_client.py --server ... --source images/scene5_reorder_close.jpg   # a photo instead

While it runs:
  * a window shows the live camera, what the VLM saw (who rests on what) and the plan;
  * frames go to the server about twice a second (POST /stream/<id>);
  * TYPE A TASK IN THE TERMINAL and press Enter -> the server plans from the latest frame;
  * keys in the window:  r = the scene changed, look again and replan
                         p = plan the last task again from scratch
                         s = save the current frame,   q / Esc = quit
  * terminal commands:   r, p, q work there too.

The live stream is also visible on the server side: http://<server-ip>:8010/camera
Headless test (no window):  --no-window --instruction "put the blue cylinder on the yellow block"
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def http(url: str, data: bytes | None = None, ctype: str = "application/json", timeout: float = 300):
    req = urllib.request.Request(url, data=data, headers={"Content-Type": ctype} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("detail")
        except Exception:                                   # noqa: BLE001
            detail = str(e)
        return {"status": "error", "message": f"HTTP {e.code}: {detail}"}
    except Exception as e:                                  # noqa: BLE001
        return {"status": "error", "message": f"{type(e).__name__}: {e}"}


class Client:
    def __init__(self, args):
        import cv2
        self.cv2, self.args = cv2, args
        self.server = args.server.rstrip("/")
        self.cam = args.camera_id
        self.frame = None
        self.frame_lock = threading.Lock()
        self.status = {"sent": 0, "last_ack": None, "error": None, "busy": None}
        self.result: dict | None = None
        self.last_instruction = args.instruction
        self.request_id = None
        self.jobs: queue.Queue = queue.Queue()
        self.stop = threading.Event()
        src = args.source
        if src is None:
            self.cap = cv2.VideoCapture(args.camera)
            if args.width:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            if not self.cap.isOpened():
                sys.exit(f"cannot open camera {args.camera} (try --camera 1; on macOS allow camera access "
                         "for the terminal in System Settings > Privacy & Security > Camera)")
            self.still = None
        elif Path(src).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            self.still, self.cap = cv2.imread(src), None
            if self.still is None:
                sys.exit(f"cannot read image {src}")
        else:
            self.cap, self.still = cv2.VideoCapture(src), None    # video file or stream URL

    # ------------------------------------------------------------ threads
    def grab(self):
        if self.still is not None:
            return self.still.copy()
        ok, img = self.cap.read()
        if not ok and self.args.source:                     # video file ended: start again
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
            ok, img = self.cap.read()
        return img if ok else None

    def sender(self):
        """Send the latest frame to the server at --fps."""
        cv2 = self.cv2
        while not self.stop.is_set():
            t0 = time.time()
            with self.frame_lock:
                img = None if self.frame is None else self.frame.copy()
            if img is not None:
                h, w = img.shape[:2]
                if self.args.send_width and w > self.args.send_width:
                    img = cv2.resize(img, (self.args.send_width, int(h * self.args.send_width / w)))
                ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.args.quality])
                if ok:
                    r = http(f"{self.server}/stream/{self.cam}", jpg.tobytes(), "image/jpeg", timeout=5)
                    if r.get("ok"):
                        self.status.update(sent=self.status["sent"] + 1, last_ack=time.time(), error=None,
                                           bytes=len(jpg))
                    else:
                        self.status["error"] = r.get("message")
            self.stop.wait(max(0.0, 1.0 / self.args.fps - (time.time() - t0)))

    def stdin_reader(self):
        print("\nType a task and press Enter (r = replan, p = plan again, q = quit):", flush=True)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if line.lower() in ("q", "quit", "exit"):
                self.stop.set()
                break
            self.jobs.put(("replan", None) if line.lower() == "r" else
                          ("plan", self.last_instruction) if line.lower() == "p" else ("plan", line))

    def worker(self):
        while not self.stop.is_set():
            try:
                kind, text = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            if kind == "plan":
                if not text:
                    print("no task yet: type one first", flush=True)
                    continue
                self.last_instruction = text
                self.status["busy"] = f"planning: {text}"
                r = http(f"{self.server}/camera/plan",
                         json.dumps({"camera_id": self.cam, "instruction": text}).encode())
            else:
                if not self.request_id:
                    print("plan first, then r replans", flush=True)
                    continue
                self.status["busy"] = "scene changed: looking again"
                r = http(f"{self.server}/camera/replan",
                         json.dumps({"camera_id": self.cam, "request_id": self.request_id}).encode())
            self.status["busy"] = None
            if r.get("request_id"):
                self.request_id = r["request_id"]
            self.result = r
            self.print_result(r)

    # ------------------------------------------------------------ output
    @staticmethod
    def seen(r: dict) -> list[str]:
        sup = (r.get("scene") or {}).get("support") or {}
        return [f"{a} on {b}" for a, b in sup.items()]

    def print_result(self, r: dict):
        print("\n" + "=" * 70)
        print("status     :", r.get("status"), "-", r.get("message") or "")
        if r.get("decision"):
            print("decision   :", r["decision"], "| changes:", r.get("changes"))
        print("seen       :", "; ".join(self.seen(r)) or "-")
        print("goal       :", r.get("goal"))
        print("constraints:", r.get("constraints"))
        for i, a in enumerate(r.get("plan") or [], 1):
            print(f"  {i:2}. {a['pddl']}")
        t = r.get("timings", {})
        print("time       :", t.get("total_s"), "s", "| request", r.get("request_id"), flush=True)

    def panel(self, h: int):
        import numpy as np
        cv2 = self.cv2
        w = 470
        img = np.full((h, w, 3), 28, np.uint8)
        y = 24

        def put(text, color=(230, 230, 230), scale=0.5, bold=1, gap=20):
            nonlocal y
            for line in textwrap.wrap(text, 52) or [""]:
                if y > h - 10:
                    return
                cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, bold, cv2.LINE_AA)
                y += gap

        ack = self.status["last_ack"]
        if self.status["error"]:
            put("server: " + self.status["error"], (80, 80, 255))
        elif ack:
            put(f"streaming to server OK  ({self.status['sent']} frames, "
                f"{self.status.get('bytes', 0) // 1024} kB, last {time.time() - ack:.1f} s ago)", (120, 220, 120))
        else:
            put("connecting to " + self.server, (0, 200, 255))
        if self.status["busy"]:
            put(self.status["busy"] + " ...", (0, 200, 255))
        put("task: " + (self.last_instruction or "(type it in the terminal)"), (255, 220, 150))
        y += 6
        r = self.result
        if r:
            put(f"status: {r.get('status')}  {r.get('decision') or ''}",
                (120, 220, 120) if r.get("status") == "ok" else (80, 80, 255))
            if r.get("message"):
                put(str(r["message"]), (80, 80, 255))
            put("VLM saw:", (180, 180, 255))
            for s in self.seen(r):
                put("  " + s, gap=18)
            if r.get("constraints"):
                put("constraints: " + json.dumps(r["constraints"]), (180, 180, 255))
            put("plan:", (180, 180, 255))
            for i, a in enumerate(r.get("plan") or [], 1):
                put(f"  {i}. {a['pddl']}", gap=18)
            if r.get("status") == "ok" and not r.get("plan"):
                put("  nothing to do")
        put("", gap=8)
        put("keys: r replan   p plan again   s save   q quit", (150, 150, 150), 0.45)
        return img

    # ------------------------------------------------------------ main loop
    def run(self):
        cv2 = self.cv2
        for fn in (self.sender, self.worker):
            threading.Thread(target=fn, daemon=True).start()
        if not self.args.no_window:
            threading.Thread(target=self.stdin_reader, daemon=True).start()
        if self.args.instruction:
            self.jobs.put(("plan", self.args.instruction))
        started, planned_once = time.time(), False
        while not self.stop.is_set():
            img = self.grab()
            if img is None:
                time.sleep(0.05)
                continue
            with self.frame_lock:
                self.frame = img
            if self.args.no_window:
                time.sleep(0.05)
                if self.result is not None and not planned_once:
                    planned_once = True
                    if self.args.replan_after:
                        time.sleep(self.args.replan_after)
                        self.result = None
                        self.jobs.put(("replan", None))
                        planned_once = "replan"
                    else:
                        break
                elif planned_once == "replan" and self.result is not None:
                    break
                if time.time() - started > self.args.timeout:
                    print("timeout waiting for the server", flush=True)
                    break
                continue
            view = img
            h = view.shape[0]
            if h > 720:
                view = cv2.resize(view, (int(view.shape[1] * 720 / h), 720))
                h = 720
            cv2.imshow("camera -> planning server", cv2.hconcat([view, self.panel(h)]))
            k = cv2.waitKey(30) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("r"):
                self.jobs.put(("replan", None))
            elif k == ord("p"):
                self.jobs.put(("plan", self.last_instruction))
            elif k == ord("s"):
                name = f"frame_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(name, img)
                print("saved", name, flush=True)
        self.stop.set()
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://localhost:8010", help="planning server URL")
    ap.add_argument("--camera", type=int, default=0, help="camera index (0, 1, ...)")
    ap.add_argument("--source", help="instead of a camera: an image, a video file or a stream URL")
    ap.add_argument("--camera-id", default="cam0", help="name of this stream on the server")
    ap.add_argument("--fps", type=float, default=2.0, help="frames sent to the server per second")
    ap.add_argument("--width", type=int, default=1280, help="requested camera width")
    ap.add_argument("--send-width", type=int, default=960, help="frames are scaled down to this width before sending")
    ap.add_argument("--quality", type=int, default=85, help="JPEG quality of sent frames")
    ap.add_argument("--instruction", help="plan this task as soon as frames arrive")
    ap.add_argument("--no-window", action="store_true", help="headless: plan --instruction once, print, exit")
    ap.add_argument("--replan-after", type=float, default=0, help="headless: replan this many seconds after the plan")
    ap.add_argument("--timeout", type=float, default=300)
    args = ap.parse_args()
    if args.no_window and not args.instruction:
        ap.error("--no-window needs --instruction")
    health = http(args.server.rstrip("/") + "/health", timeout=5)
    print("server:", health)
    if health.get("status") == "error":
        sys.exit("the server does not answer; check --server and the port forwarding")
    Client(args).run()


if __name__ == "__main__":
    main()
