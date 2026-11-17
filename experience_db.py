"""Experience database: answer from earlier requests instead of asking the model again.

Stored in one SQLite file (standard library, default logs/experience.sqlite). Three kinds:

    scene   key = model + perception prompt + token budget + the exact image bytes
            value = the model's scene answer            -> skips the perception call
    goal    key = model + the complete goal prompt
            value = the model's goal answer             -> skips the goal call
    plan    key = the exact PDDL problem + domain + planner settings
            value = the plan, the planner that found it -> skips planning

Why reuse is safe: with greedy decoding a model's answer depends only on its input. The
goal prompt contains the instruction and the list of objects and places, but NOT how they
are arranged, so the same instruction about the same objects gets the same goal, even if
the objects have been moved since. Changing the model or a prompt changes the key, so old
answers are never used for a different prompt. A cached plan is replayed on the scene
before it is returned, exactly like a new plan.

What is stored: only answers from requests that ended "ok" (a plan was found and verified).
A retry with sampling (after an unusable answer) never reads from the database.

Similar instructions (optional, --memory-examples K): when there is no exact hit, the K most
similar earlier instructions and their goals are put into the goal prompt as examples
(word overlap, no embedding model). This changes what the model sees, so it is off by default.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from pathlib import Path

KINDS = ("scene", "goal", "plan")


class ExperienceDB:
    def __init__(self, path: str | Path, model_key: str = "", examples: int = 0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model_key = model_key or ""
        self.examples = examples
        self._lock = threading.Lock()
        self.session = {k: {"hits": 0, "misses": 0} for k in KINDS}
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS entries (
                kind TEXT, key TEXT, value TEXT, meta TEXT, created REAL, last_used REAL,
                hits INTEGER DEFAULT 0, PRIMARY KEY (kind, key))""")

    def _db(self):
        return sqlite3.connect(self.path, timeout=10)

    # ------------------------------------------------------------ keys
    def key(self, kind: str, *parts) -> str:
        h = hashlib.sha256(f"{kind}|{self.model_key}".encode())
        for p in parts:
            h.update(b"\x00")
            h.update(p if isinstance(p, bytes) else str(p).encode())
        return h.hexdigest()

    # ------------------------------------------------------------ read / write
    def get(self, kind: str, key: str):
        with self._lock, self._db() as db:
            row = db.execute("SELECT value FROM entries WHERE kind=? AND key=?", (kind, key)).fetchone()
            if row:
                db.execute("UPDATE entries SET hits=hits+1, last_used=? WHERE kind=? AND key=?",
                           (time.time(), kind, key))
        self.session[kind]["hits" if row else "misses"] += 1
        return json.loads(row[0]) if row else None

    def put(self, kind: str, key: str, value, meta: dict | None = None) -> None:
        now = time.time()
        with self._lock, self._db() as db:
            db.execute("""INSERT INTO entries (kind, key, value, meta, created, last_used, hits)
                          VALUES (?, ?, ?, ?, ?, ?, 0)
                          ON CONFLICT(kind, key) DO UPDATE SET value=excluded.value, meta=excluded.meta,
                          last_used=excluded.last_used""",
                       (kind, key, json.dumps(value), json.dumps(meta or {}), now, now))

    # ------------------------------------------------------------ similar instructions
    def similar_goals(self, instruction: str, k: int) -> list[dict]:
        """The k stored goal answers whose instructions share the most words with this one."""
        if k <= 0:
            return []
        words = set(re.findall(r"\w+", instruction.lower()))
        with self._lock, self._db() as db:
            rows = db.execute("SELECT value, meta FROM entries WHERE kind='goal'").fetchall()
        scored = []
        for value, meta in rows:
            m = json.loads(meta)
            other = set(re.findall(r"\w+", m.get("instruction", "").lower()))
            if not other or m.get("instruction", "").strip().lower() == instruction.strip().lower():
                continue
            score = len(words & other) / (len(words | other) or 1)
            if score > 0:
                scored.append((score, {"instruction": m["instruction"], "objects": m.get("objects", ""),
                                       "answer": json.loads(value)}))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:k]]

    def examples_block(self, instruction: str) -> str:
        ex = self.similar_goals(instruction, self.examples)
        if not ex:
            return ""
        lines = ["Earlier instructions in other scenes and the answers that worked (ids differ):"]
        for e in ex:
            lines.append(f"- objects: {e['objects']}\n  instruction: {e['instruction']}\n  answer: "
                         f"{e['answer'].strip()[:600]}")
        return "\n".join(lines) + "\n\n"

    # ------------------------------------------------------------ bookkeeping
    def stats(self) -> dict:
        with self._lock, self._db() as db:
            rows = db.execute("SELECT kind, COUNT(*), COALESCE(SUM(hits), 0) FROM entries GROUP BY kind").fetchall()
        stored = {k: {"entries": 0, "hits_all_time": 0} for k in KINDS}
        for kind, n, hits in rows:
            stored[kind] = {"entries": n, "hits_all_time": hits}
        return {"path": str(self.path), "model": self.model_key, "examples": self.examples,
                "bytes": self.path.stat().st_size if self.path.exists() else 0,
                "stored": stored, "this_run": self.session}

    def clear(self, kind: str | None = None) -> int:
        with self._lock, self._db() as db:
            cur = db.execute("DELETE FROM entries" + (" WHERE kind=?" if kind else ""), (kind,) if kind else ())
            return cur.rowcount
