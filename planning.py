"""PDDL parsing, consistency checks, planning and plan validation (unified-planning)."""
from __future__ import annotations

import re
import tempfile
import time
import warnings
from pathlib import Path

from unified_planning.engines.sequential_simulator import UPSequentialSimulator
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner, get_environment

from scenes import DOMAIN_PATH

get_environment().credits_stream = None      # silence the planner banners
warnings.filterwarnings('ignore', message='Pyperplan does not support timeout')
SUPPORT_PREDICATES = ('on', 'on-table', 'in')
PLANNERS = ('fast-downward', 'pyperplan')   # 'auto' tries these in order
# also usable with --planner: 'fast-downward-opt' (optimal, slower), 'pyperplan'


# ---------------------------------------------------------------- extraction from model text

def strip_reasoning(text: str) -> str:
    """Remove <think>...</think>; prefer the <answer> block if the model used one."""
    m = re.search(r'<answer>(.*?)(</answer>|$)', text, re.S)
    if m:
        return m.group(1).strip()
    if '</think>' in text:
        return text.rsplit('</think>', 1)[1].strip()
    return text.strip()


def _balanced(text: str, start: int, open_c: str, close_c: str) -> str | None:
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_c:
            depth += 1
        elif text[i] == close_c:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def extract_problem(text: str) -> str | None:
    text = re.sub(r';[^\n]*', '', strip_reasoning(text))
    m = re.search(r'\(\s*define\s*\(\s*problem', text, re.I)
    return _balanced(text, m.start(), '(', ')') if m else None


def extract_json(text: str) -> str | None:
    text = strip_reasoning(text)
    i = text.find('{')
    return _balanced(text, i, '{', '}') if i >= 0 else None


def extract_plan(text: str) -> list[tuple[str, list[str]]]:
    """Action lines like '(pick-from-item a b)' or 'pick-from-item(a, b)'."""
    text = strip_reasoning(text)
    func = re.compile(r'([a-z][a-z0-9_-]*)\(([^()]*)\)', re.I)              # name(a, b)
    pddl = re.compile(r'\(\s*([a-z][a-z0-9_-]*)((?:\s+[a-z0-9_]+)*)\s*\)', re.I)  # (name a b)
    steps = []
    for line in text.splitlines():
        m = func.search(line)
        if m:
            args = [a.strip().strip('\'"') for a in m.group(2).split(',') if a.strip()]
            steps.append((m.group(1).lower(), args))
            continue
        for m in pddl.finditer(line):
            steps.append((m.group(1).lower(), m.group(2).split()))
    return steps


# ---------------------------------------------------------------- parsing

def parse_problem(problem_text: str, domain_path: Path = DOMAIN_PATH):
    """Return a unified-planning Problem; raises on syntax / type / unknown-name errors."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'problem.pddl'
        p.write_text(problem_text)
        return PDDLReader().parse_problem(str(domain_path), str(p))


def _atom(f) -> tuple[str, ...]:
    return (f.fluent().name, *[str(a) for a in f.args])


def init_facts(problem) -> set[tuple[str, ...]]:
    return {_atom(k) for k, v in problem.explicit_initial_values.items()
            if v.is_bool_constant() and v.bool_constant_value()}


def goal_facts(problem) -> set[tuple[str, ...]]:
    out, stack = set(), list(problem.goals)
    while stack:
        g = stack.pop()
        if g.is_and():
            stack.extend(g.args)
        elif g.is_fluent_exp():
            out.add(_atom(g))
        else:
            out.add(('<complex>', str(g)))
    return out


def lint_state(problem) -> list[str]:
    """Physical consistency of the initial state (things the parser cannot catch)."""
    facts = init_facts(problem)
    items = [str(o) for o in problem.objects(problem.user_type('item'))] \
        if problem.has_type('item') else []
    issues = []
    held = [f[1] for f in facts if f[0] == 'holding']
    for x in items:
        n = sum(1 for f in facts if f[0] in SUPPORT_PREDICATES and f[1] == x) + (x in held)
        if n != 1:
            issues.append(f'{x} has {n} supports (needs exactly 1)')
    covered = {f[2] for f in facts if f[0] == 'on'}
    for x in items:
        is_clear = ('clear', x) in facts
        should = x not in covered and x not in held
        if is_clear != should:
            issues.append(f'clear({x}) is {is_clear} but should be {should}')
    if (('hand-empty',) in facts) == bool(held):
        issues.append('hand-empty inconsistent with holding')
    for f in facts:
        if f[0] == 'in' and not any(g[0] == 'occupies' and g[1] == f[1] for g in facts):
            issues.append(f'{f[1]} is in {f[2]} but occupies no slot')
        if f[0] == 'occupies' and ('slot-free', f[2]) in facts:
            issues.append(f'slot {f[2]} is both occupied and free')
    return issues


# ---------------------------------------------------------------- planning

def solve(problem, planner: str = 'auto', timeout: float = 60):
    """Return (status, actions as [(name, [args])], seconds, planner_name)."""
    names = PLANNERS if planner == 'auto' else (planner,)
    last_err = None
    for name in names:
        try:
            t0 = time.perf_counter()
            with OneshotPlanner(name=name) as engine:
                res = engine.solve(problem, timeout=timeout)
            dt = time.perf_counter() - t0
        except Exception as e:                      # planner not installed / unsupported
            last_err = e
            continue
        status = res.status.name.lower()
        if res.plan is None:
            return status, None, dt, name
        acts = [(a.action.name, [str(p) for p in a.actual_parameters]) for a in res.plan.actions]
        return status, acts, dt, name
    raise RuntimeError(f'no planner available: {last_err}')


def simulate(problem, steps: list[tuple[str, list[str]]]) -> dict:
    """Execute a plan step by step on `problem` (usually the ground-truth problem).

    Returns ok (all steps applicable AND goal reached), executed steps, first failure and reason.
    """
    out = {'ok': False, 'steps': len(steps), 'executed': 0, 'fail_step': None,
           'fail_reason': None, 'goal_reached': False}
    with UPSequentialSimulator(problem, error_on_failed_checks=False) as sim:
        state = sim.get_initial_state()
        for k, (name, args) in enumerate(steps):
            try:
                action = problem.action(name)
            except Exception:
                out.update(fail_step=k, fail_reason=f'unknown action {name}')
                return out
            if len(args) != len(action.parameters):
                out.update(fail_step=k, fail_reason=f'{name} expects {len(action.parameters)} args')
                return out
            try:
                objs = [problem.object(a) for a in args]
            except Exception:
                out.update(fail_step=k, fail_reason=f'unknown object in {name} {args}')
                return out
            try:
                applicable = sim.is_applicable(state, action, objs)
            except Exception as e:                  # wrong types
                out.update(fail_step=k, fail_reason=f'bad parameters {name} {args}: {e}')
                return out
            if not applicable:
                out.update(fail_step=k, fail_reason=f'precondition violated: {name} {" ".join(args)}')
                return out
            state = sim.apply(state, action, objs)
            out['executed'] = k + 1
        out['goal_reached'] = bool(sim.is_goal(state))
        out['ok'] = out['goal_reached']
        if not out['goal_reached']:
            out['fail_reason'] = 'goal not reached'
    return out


def prf(pred: set, gold: set) -> dict:
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 1.0
    return {'precision': round(p, 3), 'recall': round(r, 3),
            'f1': round(2 * p * r / (p + r), 3) if p + r else 0.0}
