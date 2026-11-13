"""A learned next-move policy with a symbolic safety net (architecture I).

Shape of the approach: generate scenes -> solve them optimally with A* -> train a
model to rank the legal moves of a state so that the optimal move comes first ->
at run time roll the model out greedily -> validate -> fall back to the symbolic
planner if the rollout is not a valid plan.

To keep the example dependency-light the model here is a linear ranker over a
few hand-made relational features, trained with numpy. A GNN would replace
`features()` by message passing over the scene graph (objects and locations as
nodes, supported-by / goal edges) and score candidate moves from node
embeddings; training data, rollout, validation and fallback stay the same.
"""
from __future__ import annotations

import random

from .base import Proposal, Strategy
from .symbolic import GraphStrategy
from semplan.gen import random_scene
from semplan.planner import plan_astar, quick_infeasible
from semplan.scene import Move, Scene, apply, legal_dests, validate

N_FEATURES = 12


def candidates(s: Scene) -> list[Move]:
    held = s.held()
    movers = [held] if held else sorted(o for o in s.clear() if s.movable(o))
    return [Move(o, d) for o in movers for d in legal_dests(s, o) if d != s.on[o]]


def features(s: Scene, m: Move, goals: dict, settled: set) -> list[float]:
    """Relational features of doing move m in state s."""
    o, d = m.obj, m.dst
    g = goals.get(o)
    dst_is_obj = d in s.objects
    dst_region = s.base(d) if dst_is_obj else d
    below = s.on[o]
    kind = "object" if dst_is_obj else s.locs[d].kind
    waiting = sum(1 for x, z in goals.items() if z == g and x not in settled and x != o)
    return [
        1.0,
        float(o in settled),                                   # moving a finished object
        float(g is not None and dst_region == g),              # ends in its goal zone
        float(g is not None and dst_region == g and (not dst_is_obj or d in settled)),  # final
        float(kind == "buffer"),
        float(kind == "zone" and dst_region != g),             # borrows another zone
        float(dst_is_obj and d not in settled),                # onto something that must move
        float(below in s.objects and below not in settled),    # uncovers an unfinished object
        float(below in s.locs and s.locs[below].kind == "buffer"),   # leaves a parking slot
        float(not s.objects[o].can_support),                   # a prism
        float(not s.objects[o].can_support and kind == "zone" and waiting > 0),
        float(g is None),
    ]


def make_dataset(n_scenes: int, seed: int = 0, sizes=(3, 6)):
    """(feature rows of all candidates, index of the optimal one) for states on A* plans."""
    rng = random.Random(seed)
    data = []
    while len(data) < n_scenes:
        s = random_scene(rng, rng.randint(*sizes), n_colors=3, n_buffers=rng.choice([0, 1, 2]))
        if quick_infeasible(s):
            continue
        r = plan_astar(s, 100_000, 2_000)
        if r.status != "ok" or not r.moves:
            continue
        s = s.copy()
        steps = []
        for m in r.moves:
            cands = candidates(s)
            if m in cands:
                goals = s.goals()
                settled = s.settled(goals)
                steps.append(([features(s, c, goals, settled) for c in cands], cands.index(m)))
            apply(s, m)
        data.append(steps)
    return data


def train(data, epochs: int = 60, lr: float = 0.5, seed: int = 0):
    """Softmax ranking loss over the candidate moves of each state (plain gradient descent)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, N_FEATURES)
    states = [(np.asarray(x, dtype=float), y) for steps in data for x, y in steps]
    for _ in range(epochs):
        grad = np.zeros(N_FEATURES)
        for x, y in states:
            z = x @ w
            p = np.exp(z - z.max())
            p /= p.sum()
            grad += x.T @ p - x[y]
        w -= lr * grad / len(states)
    acc = sum(int((x @ w).argmax() == y) for x, y in states) / len(states)
    return w.tolist(), acc


class LearnedPolicyStrategy:
    """Greedy rollout of the learned ranker; the symbolic planner is the safety net."""

    def __init__(self, weights: list[float], fallback: Strategy | None = GraphStrategy()):
        self.name = "learned+symbolic" if fallback else "learned"
        self.w, self.fallback = weights, fallback
        self.used_fallback = 0

    def rollout(self, scene: Scene) -> list[Move] | None:
        s = scene.copy()
        moves, seen = [], {tuple(sorted(s.on.items()))}
        for _ in range(3 * len(s.objects) + 6):
            if s.goal_reached():
                return moves
            cands = candidates(s)
            if not cands:
                return None
            goals = s.goals()
            settled = s.settled(goals)
            best = max(cands, key=lambda c: sum(a * b for a, b in
                                                zip(self.w, features(s, c, goals, settled))))
            apply(s, best)
            key = tuple(sorted(s.on.items()))
            if key in seen:
                return None                          # the policy is going in circles
            seen.add(key)
            moves.append(best)
        return moves if s.goal_reached() else None

    def propose(self, scene: Scene) -> Proposal:
        moves = self.rollout(scene)
        if moves is not None and validate(scene, moves).ok:
            return Proposal(moves, source="learned")
        if self.fallback is None:
            return Proposal(None, "learned policy did not reach the goal", "learned")
        self.used_fallback += 1
        p = self.fallback.propose(scene)
        p.source = f"learned failed -> {p.source}"
        return p
