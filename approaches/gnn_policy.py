"""A real graph neural network policy, trained and measured (needs `pip install torch`).

    python -m approaches.gnn_policy            # trains, evaluates, writes results/gnn_bench.json

Graph of a scene
    nodes      every object and every location (zones, free slots, table)
    edges      on (object -> its support), carries (reverse), goal (object -> its goal zone),
               wanted_by (reverse)
    features   node kind, can it carry, is it clear, is it final, is its goal met, free capacity, height

Model         3 rounds of relational message passing (one weight matrix per edge type, sum over
              neighbours), then a small network scores each legal move from the embeddings of the
              moved object and the destination.
Training      imitation of optimal A* plans: for every state on a plan, the optimal move should get
              the highest score among the legal moves (cross-entropy).
Use           greedy rollout -> validator -> symbolic planner as fallback, exactly like the linear
              ranker in learned_policy.py.

The point of this file is the comparison at the bottom: does a GNN plan better or faster than
the symbolic planner it learns from?
"""
from __future__ import annotations

import json
import platform
import random
import time
from pathlib import Path

import torch
import torch.nn as nn

from . import run
from .base import Proposal, Strategy
from .learned_policy import LearnedPolicyStrategy, candidates, make_dataset, train as train_linear
from .symbolic import GraphStrategy
from semplan.gen import random_scene
from semplan.planner import plan_astar, quick_infeasible
from semplan.scene import HELD, Move, Scene, apply, validate

OUT = Path(__file__).parent / "results" / "gnn_bench.json"
RELATIONS = ("on", "carries", "goal", "wanted_by")
N_FEAT = 12


def encode(s: Scene):
    """Scene -> (node feature matrix, {relation: (src index tensor, dst index tensor)}, node index)."""
    goals = s.goals()
    settled, clear = s.settled(goals), s.clear()
    names = sorted(s.objects) + sorted(s.locs)
    idx = {n: i for i, n in enumerate(names)}
    x = torch.zeros(len(names), N_FEAT)
    for o, ob in s.objects.items():
        i = idx[o]
        height, cur = 0, s.on[o]
        while cur in s.objects:
            height, cur = height + 1, s.on[cur]
        g = goals.get(o)
        x[i, 0] = 1.0
        x[i, 4] = float(ob.can_support)
        x[i, 5] = float(o in clear)
        x[i, 6] = float(o in settled)
        x[i, 7] = float(g is not None)
        x[i, 8] = float(g is not None and s.base(o) == g)
        x[i, 9] = min(height, 4) / 4
        x[i, 10] = float(s.movable(o))
    for l, loc in s.locs.items():
        i = idx[l]
        x[i, {"zone": 1, "buffer": 2, "table": 3}[loc.kind]] = 1.0
        if loc.kind != "table":
            x[i, 11] = min(s.floor_free(l), 3) / 3
    rel = {r: ([], []) for r in RELATIONS}
    for o, sup in s.on.items():
        if sup != HELD:
            rel["on"][0].append(idx[o]); rel["on"][1].append(idx[sup])
            rel["carries"][0].append(idx[sup]); rel["carries"][1].append(idx[o])
    for o, z in goals.items():
        rel["goal"][0].append(idx[o]); rel["goal"][1].append(idx[z])
        rel["wanted_by"][0].append(idx[z]); rel["wanted_by"][1].append(idx[o])
    edges = {r: (torch.tensor(a, dtype=torch.long), torch.tensor(b, dtype=torch.long)) for r, (a, b) in rel.items()}
    return x, edges, idx


class SceneGNN(nn.Module):
    def __init__(self, hidden: int = 48, rounds: int = 3):
        super().__init__()
        self.inp = nn.Linear(N_FEAT, hidden)
        self.self_w = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(rounds))
        self.rel_w = nn.ModuleList(nn.ModuleDict({r: nn.Linear(hidden, hidden, bias=False) for r in RELATIONS})
                                   for _ in range(rounds))
        self.score = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def embed(self, x, edges):
        h = torch.relu(self.inp(x))
        for self_w, rel_w in zip(self.self_w, self.rel_w):
            out = self_w(h)
            for r, (src, dst) in edges.items():
                if len(src):
                    out = out.index_add(0, dst, rel_w[r](h[src]))     # message src -> dst
            h = torch.relu(out)
        return h

    def forward(self, x, edges, cand_obj, cand_dst):
        h = self.embed(x, edges)
        return self.score(torch.cat([h[cand_obj], h[cand_dst]], dim=1)).squeeze(1)


def graph_dataset(n_scenes: int, seed: int, sizes=(3, 6)):
    """[(x, edges, candidate object idx, candidate destination idx, label)] from optimal plans."""
    rng = random.Random(seed)
    data, scenes = [], 0
    while scenes < n_scenes:
        s = random_scene(rng, rng.randint(*sizes), n_colors=3, n_buffers=rng.choice([0, 1, 2]))
        if quick_infeasible(s):
            continue
        r = plan_astar(s, 100_000, 2_000)
        if r.status != "ok" or not r.moves:
            continue
        scenes += 1
        s = s.copy()
        for m in r.moves:
            cands = candidates(s)
            if m in cands:
                x, edges, idx = encode(s)
                data.append((x, edges, torch.tensor([idx[c.obj] for c in cands]),
                             torch.tensor([idx[c.dst] for c in cands]), cands.index(m)))
            apply(s, m)
    return data


def train_gnn(data, epochs: int = 25, lr: float = 3e-3, seed: int = 0, batch: int = 32):
    torch.manual_seed(seed)
    model = SceneGNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed)
    for _ in range(epochs):
        rng.shuffle(data)
        for k in range(0, len(data), batch):
            loss = 0.0
            for x, edges, co, cd, y in data[k:k + batch]:
                logits = model(x, edges, co, cd)
                loss = loss + nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([y]))
            opt.zero_grad()
            (loss / batch).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        acc = sum(int(model(x, e, co, cd).argmax()) == y for x, e, co, cd, y in data) / len(data)
    return model, acc


class GnnPolicyStrategy:
    """Greedy rollout of the GNN; optional symbolic fallback when the rollout is not a valid plan."""

    def __init__(self, model: SceneGNN, fallback: Strategy | None = None):
        self.name = "gnn+symbolic" if fallback else "gnn"
        self.model, self.fallback = model, fallback
        self.used_fallback = 0

    @torch.no_grad()
    def rollout(self, scene: Scene) -> list[Move] | None:
        s = scene.copy()
        moves, seen = [], {tuple(sorted(s.on.items()))}
        for _ in range(3 * len(s.objects) + 6):
            if s.goal_reached():
                return moves
            cands = candidates(s)
            if not cands:
                return None
            x, edges, idx = encode(s)
            scores = self.model(x, edges, torch.tensor([idx[c.obj] for c in cands]),
                                torch.tensor([idx[c.dst] for c in cands]))
            best = cands[int(scores.argmax())]
            apply(s, best)
            key = tuple(sorted(s.on.items()))
            if key in seen:
                return None
            seen.add(key)
            moves.append(best)
        return moves if s.goal_reached() else None

    def propose(self, scene: Scene) -> Proposal:
        moves = self.rollout(scene)
        if moves is not None and validate(scene, moves).ok:
            return Proposal(moves, source="gnn")
        if self.fallback is None:
            return Proposal(None, "GNN rollout did not reach the goal", "gnn")
        self.used_fallback += 1
        p = self.fallback.propose(scene)
        p.source = f"gnn failed -> {p.source}"
        return p


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def main():
    torch.set_num_threads(1)                         # comparable with the single-threaded planners
    res = {"meta": {"system": platform.platform(), "torch": torch.__version__,
                    "date": time.strftime("%Y-%m-%d")}}
    sizes = {}
    for n_train in (500, 2000):
        t0 = time.perf_counter()
        data = graph_dataset(n_train, seed=5)
        t1 = time.perf_counter()
        model, acc = train_gnn(list(data))
        t2 = time.perf_counter()
        params = sum(p.numel() for p in model.parameters())
        lin_w, lin_acc = train_linear(make_dataset(n_train, seed=5))
        print(f"train scenes {n_train}: {len(data)} states, GNN top-1 {acc:.3f} ({params} parameters, "
              f"data {t1 - t0:.0f} s, training {t2 - t1:.0f} s); linear top-1 {lin_acc:.3f}", flush=True)
        entry = {"train_states": len(data), "gnn_top1": acc, "linear_top1": lin_acc, "parameters": params,
                 "data_s": t1 - t0, "train_s": t2 - t1, "eval": {}}
        rng = random.Random(77)
        for label, (lo, hi) in (("3-6 objects (as in training)", (3, 6)), ("7-10 objects (larger than training)", (7, 10))):
            scenes = []
            while len(scenes) < 300:
                s = random_scene(rng, rng.randint(lo, hi), n_colors=3, n_buffers=rng.choice([0, 1, 2]))
                if not quick_infeasible(s) and run(GraphStrategy(), s).status == "ok":
                    scenes.append(s)
            rows = {}
            for name, stg in (("graph planner", GraphStrategy()),
                              ("linear ranker", LearnedPolicyStrategy(lin_w, None)),
                              ("GNN", GnnPolicyStrategy(model)),
                              ("GNN + symbolic fallback", GnnPolicyStrategy(model, GraphStrategy()))):
                ms, ok, moves, ref = [], 0, 0, 0
                for s in scenes:
                    o = run(stg, s)
                    ms.append(o.ms)
                    if o.valid:
                        ok += 1
                        moves += len(o.moves)
                        ref += len(run(GraphStrategy(), s).moves)
                rows[name] = {"valid": ok, "of": len(scenes), "p50_ms": pct(ms, .5), "p95_ms": pct(ms, .95),
                              "moves_vs_graph_planner": moves / ref if ref else None}
                print(f"  {label:<36} {name:<26} valid {ok}/{len(scenes)}  p50 {pct(ms, .5):.3f} ms  "
                      f"p95 {pct(ms, .95):.3f} ms  moves vs planner {moves / ref:.3f}", flush=True)
            entry["eval"][label] = rows
        sizes[str(n_train)] = entry
    res["by_training_size"] = sizes
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
