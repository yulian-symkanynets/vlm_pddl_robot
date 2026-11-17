"""Start moving before the model has finished: release streamed moves through a gate.

When a model writes the plan (the vlm_plan mode), its answer arrives token by token.
Waiting for the whole answer wastes time: the robot could already execute the first
move. But a move that is released cannot be taken back, so a wrong early move costs
real robot time (the "blue on red, then blue has to move again" problem).

Every complete MOVE line is therefore checked before it is released:

    gate "none"      release every line at once (the risky version of the idea)
    gate "legal"     release a move if its preconditions hold in the current state
    gate "optimal"   release a move only if it lies on a SHORTEST plan from the current
                     state: optimal_length(after) == optimal_length(before) - 1,
                     computed with the A* planner (milliseconds for small scenes)

The first move that fails the gate ends the stream (the generation can be cancelled)
and the symbolic planner finishes the plan from the state reached so far. With the
"optimal" gate the total number of moves is never larger than the optimal number,
however many mistakes the model makes, and every move the model got right is
executed while the model is still writing.

`simulate_stream` replays a model answer at a given token rate, so the effect on
time-to-first-move can be measured without a GPU. `stream_from_model` connects the
same logic to a real model (vlm.HFVLM.stream).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from semplan.planner import plan, plan_astar
from semplan.scene import Move, Scene, apply, validate, why_not
from semplan.task_planner import plan_task

MOVE_LINE = re.compile(r"MOVE[\s(]+([\w-]+)[\s,>\-]+([\w-]+)", re.I)


def optimal_length(scene: Scene, budget_ms: float = 200.0) -> int | None:
    if scene.general:
        r = plan_task(scene, budget_ms)
        return len(r.moves) if r.status in ("ok", "partial") and r.planner == "task_astar" else None
    r = plan_astar(scene, 10 ** 9, budget_ms)
    return len(r.moves) if r.status == "ok" else None


@dataclass
class Release:
    move: Move
    at_token: int                      # the move became available after this many output tokens
    gate: str                          # what the move passed


@dataclass
class GatedResult:
    released: list[Release] = field(default_factory=list)
    completion: list[Move] = field(default_factory=list)   # planned by the server after the stop
    stopped_at_token: int | None = None                     # None: the model's answer was used to the end
    stop_reason: str = ""
    tokens_read: int = 0

    @property
    def moves(self) -> list[Move]:
        return [r.move for r in self.released] + self.completion


def gated(scene: Scene, chunks: Iterable[str], gate: str = "optimal") -> Iterator[Release | GatedResult]:
    """Consume text chunks (one or more tokens each). Yields a Release as soon as a move is
    accepted, and finally a GatedResult with the complete plan."""
    state = scene.copy()
    res = GatedResult()
    buf, tokens = "", 0
    best = optimal_length(state) if gate == "optimal" else None
    for chunk in chunks:
        tokens += 1
        buf += chunk
        if "</think>" in buf:                       # only the answer counts, not the reasoning
            buf = buf.rsplit("</think>", 1)[1]
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            m = MOVE_LINE.search(line)
            if not m:
                continue
            move = Move(m.group(1), m.group(2))
            reason = None if gate == "none" else why_not(state, move)
            if reason is None and gate == "optimal":
                after = state.copy()
                apply(after, move)
                opt_after = optimal_length(after)
                if best is None or opt_after is None or opt_after != best - 1:
                    reason = "not on a shortest plan (it would cost extra moves later)"
                else:
                    best = opt_after
            if reason is not None:
                res.stopped_at_token, res.stop_reason = tokens, f"{move}: {reason}"
                res.tokens_read = tokens
                res.completion = _finish(state)
                yield res
                return
            if gate == "none" and why_not(state, move) is not None:
                res.released.append(Release(move, tokens, "none (illegal)"))
                continue                            # the robot would fail here; state unchanged
            apply(state, move)
            rel = Release(move, tokens, gate)
            res.released.append(rel)
            yield rel
    res.tokens_read = tokens
    if not state.goal_reached():
        res.stop_reason = res.stop_reason or "the model's plan ended before the goal"
        res.completion = _finish(state)
    yield res


def _finish(state: Scene) -> list[Move]:
    """Plan the rest from the state reached: shortest plan if it can be found in time."""
    if state.general:
        r = plan_task(state, 300)
    else:
        r = plan(state, optimal=True, astar_ms=200)
        if r.status not in ("ok", "partial"):
            r = plan(state, astar_ms=100)
    return r.moves if r.status in ("ok", "partial") else []


# ================================================================ measurement without a GPU

def tokenize(answer: str, thinking_tokens: int = 0) -> list[str]:
    """Rough token stream: an optional reasoning block, then the answer word by word."""
    think = ["<think>"] + ["…"] * thinking_tokens + ["</think>\n"] if thinking_tokens else []
    out = []
    for line in answer.splitlines():
        words = line.split(" ")
        out += [w + " " for w in words[:-1]] + [words[-1] + "\n"]
    return think + out


def simulate_stream(scene: Scene, answer: str, gate: str, tokens_per_s: float = 25.0,
                    thinking_tokens: int = 0) -> dict:
    """Replay a model answer at a fixed token rate. Times are in seconds after the request."""
    chunks = tokenize(answer, thinking_tokens)
    result = None
    first = None
    for ev in gated(scene, chunks, gate):
        if isinstance(ev, Release) and first is None:
            first = ev.at_token / tokens_per_s
        if isinstance(ev, GatedResult):
            result = ev
    full_answer_s = len(chunks) / tokens_per_s
    stop_s = (result.stopped_at_token or len(chunks)) / tokens_per_s
    valid = validate(scene, result.moves).ok
    return {"first_move_s": first if first is not None else stop_s,
            "plan_complete_s": stop_s,
            "wait_for_full_answer_s": full_answer_s,
            "moves": len(result.moves), "valid": valid,
            "released_by_model": sum(1 for r in result.released if r.gate != "none (illegal)"),
            "illegal_released": sum(1 for r in result.released if r.gate == "none (illegal)"),
            "stop_reason": result.stop_reason}


def stream_from_model(model, scene: Scene, prompt: str, gate: str = "optimal") -> Iterator[Release | GatedResult]:
    """Real model: `model` is a vlm.HFVLM (or anything with .stream(images, prompt))."""
    yield from gated(scene, model.stream([], prompt), gate)


def experiment(n: int = 200, seed: int = 0, error_rate: float = 0.4, tokens_per_s: float = 25.0) -> dict:
    """Random scenes; the 'model' answers with an optimal plan, but with probability
    `error_rate` one of its moves is replaced by a legal move that is not on any shortest
    plan (a wasted move). Compares gates for a model with and without a reasoning phase."""
    import random as _r

    from semplan.gen import random_scene
    from semplan.planner import quick_infeasible
    from semplan.scene import legal_dests
    rng = _r.Random(seed)
    cases = []
    while len(cases) < n:
        s = random_scene(rng, rng.randint(4, 7), n_colors=3, n_buffers=rng.choice([1, 2]))
        if quick_infeasible(s):
            continue
        opt = plan_astar(s, 10 ** 9, 200)
        if opt.status != "ok" or len(opt.moves) < 3:
            continue
        moves = list(opt.moves)
        if rng.random() < error_rate:                # inject one legal, wasteful move
            k = rng.randrange(len(moves))
            st = s.copy()
            for m in moves[:k]:
                apply(st, m)
            best = optimal_length(st)
            bad = []
            for o in sorted(st.clear()):
                for d in legal_dests(st, o):
                    if d == st.on[o]:
                        continue
                    after = st.copy()
                    apply(after, Move(o, d))
                    if optimal_length(after) not in (None, best - 1):
                        bad.append(Move(o, d))
            if bad:
                moves[k] = rng.choice(bad)           # the rest of the answer no longer fits
        cases.append((s, len(opt.moves), "\n".join(f"MOVE {m.obj} {m.dst}" for m in moves)))
    out = {}
    for thinking in (0, 400):
        for gate in ("none", "legal", "optimal"):
            rows = [simulate_stream(s, ans, gate, tokens_per_s, thinking) for s, _, ans in cases]
            extra = [r["moves"] - opt for r, (_, opt, _) in zip(rows, cases) if r["valid"]]
            out[f"thinking={thinking} gate={gate}"] = {
                "valid": sum(r["valid"] for r in rows), "of": n,
                "extra_moves_total": sum(extra), "plans_with_extra_moves": sum(1 for e in extra if e > 0),
                "first_move_s": sum(r["first_move_s"] for r in rows) / n,
                "plan_complete_s": sum(r["plan_complete_s"] for r in rows) / n,
                "wait_for_full_answer_s": sum(r["wait_for_full_answer_s"] for r in rows) / n,
                "illegal_moves_sent": sum(r["illegal_released"] for r in rows)}
    return out


if __name__ == "__main__":
    import json
    from pathlib import Path
    res = experiment()
    for k, v in res.items():
        print(f"{k:<28} valid {v['valid']}/{v['of']}  extra moves {v['extra_moves_total']:>3} "
              f"(in {v['plans_with_extra_moves']} plans)  illegal sent {v['illegal_moves_sent']:>3}  "
              f"first move {v['first_move_s']:5.1f} s  plan done {v['plan_complete_s']:5.1f} s  "
              f"(waiting for the whole answer: {v['wait_for_full_answer_s']:5.1f} s)")
    Path(__file__).with_name("results").joinpath("speculative.json").write_text(json.dumps(res, indent=1))
