"""The MODEL writes the moves; a gate lets only safe moves through; the planner finishes the rest.

Used by POST /plan/model (the other endpoints are unchanged). The brief asks that the VLM proposes
the sequence of operations and that the plan is checked before execution. Small models make wrong
moves, so no model move reaches the robot unchecked:

    1  the model gets the current arrangement, the goal, the rules (and the photo, if one was sent)
       and writes moves: [["object", "table | object | zone"], ...]
    2  the gate checks each move in order with exactly the planner's rules (formula_plan.explain):
         gate "legal"    the move is allowed and does not return to an arrangement already reached
         gate "optimal"  in addition the move lies on a shortest plan (the optimum shrinks by one)
       The first move that fails stops the model's plan.
    3  repair: the model is told why that move was rejected and writes the remaining moves from the
       arrangement reached so far (`repairs` rounds)
    4  the planner finishes from wherever the model stopped, and the whole plan goes through the same
       three checks as every plan (robot rules, scene limits, goal reached)

The answer reports how many moves came from the model and how many from the planner, every rejected
move with its reason, and the comparison with the planner's own plan (the measure the brief asks for:
which models generate correct sequences, and how far they are from the shortest plan).
"""
from __future__ import annotations

import json
import time

GATES = ('legal', 'optimal')


# --------------------------------------------------------------------------- words for the prompt
def goal_words(sc: dict, grounded) -> str:
    import goal_logic
    if grounded and grounded[0][0] == 'zstack':
        zone_of = json.loads(grounded[0][1])
        return ('one tower per colour, standing in the zone of that colour (' +
                ', '.join(f'{c} -> {z}' for c, z in sorted(zone_of.items())) +
                '): every object of a colour ends in that tower, exactly one object of the colour stands '
                'directly in the zone, the others on objects of the same colour')
    if grounded and all(g[0] == 'col-ok' for g in grounded):
        return 'one stack per colour anywhere on the table: all objects of a colour in one stack'
    if grounded and grounded[0][0] == 'formula':
        return 'at the end this must be true: ' + goal_logic.describe(json.loads(grounded[0][1]))
    parts = []
    for g in grounded:
        if g[0] == 'on-table':
            parts.append(f'{g[1]} on the table')
        elif g[0] == 'in':
            parts.append(f'{g[1]} directly in {g[2]}')
        else:
            parts.append(f'{g[1]} directly on {g[2]}')
    return '; '.join(parts)


def rule_words(sc: dict, cons: dict) -> list[str]:
    out = []
    if cons.get('no_table'):
        out.append('the table may not be used')
    if cons.get('keep'):
        out.append('do not move: ' + ', '.join(cons['keep']))
    for a, t in cons.get('never_on') or []:
        out.append(f'never put {"any object" if a == "*" else a} on {t}')
    if cons.get('first'):
        f = cons['first'][0] if isinstance(cons['first'], list) else cons['first']
        out.append(f'the first move must move {f}')
    if cons.get('max_stack_height'):
        out.append(f'at most {cons["max_stack_height"]} objects in a tower')
    rules = sc.get('rules') or {}
    if rules.get('no_larger_on_smaller') and any('size_mm' in d for d in sc['items'].values()):
        out.append('a larger object (size_mm) may never stand on a smaller one')
    if rules.get('max_height_mm'):
        out.append(f'no tower taller than {rules["max_height_mm"]:g} mm')
    if rules.get('max_objects'):
        out.append(f'at most {rules["max_objects"]} objects in a tower')
    for p, d in sc['env']['places'].items():
        if d.get('max_height_mm'):
            out.append(f'the tower in {p} may be at most {d["max_height_mm"]:g} mm tall')
        if d.get('max_objects'):
            out.append(f'at most {d["max_objects"]} objects in a tower in {p}')
    return out


def describe_state(sc: dict, sup: dict) -> str:
    items, places = sc['items'], sc['env']['places']
    lines = []
    for x, d in items.items():
        extra = [f'{d["size_mm"]:g} mm wide' if 'size_mm' in d else '', f'{d["height_mm"]:g} mm tall'
                 if 'height_mm' in d else '']
        extra = ', '.join(e for e in extra if e)
        top = next((y for y, v in sup.items() if v == x), None)
        where = 'the table' if sup.get(x) == 'table' else sup.get(x)
        lines.append(f'- {x} ({d.get("label", x)}{", " + extra if extra else ""}) stands on {where}; '
                     + (f'{top} is on it' if top else 'nothing is on it'))
    zones = []
    for p, d in places.items():
        used = sum(1 for v in sup.values() if v == p)
        zones.append(f'- {p} ({d.get("color", d["kind"])}, {used} of {d["slots"]} slots used)')
    return '\n'.join(lines), '\n'.join(zones) or '- (none)'


def prompt(sc: dict, sup: dict, grounded, cons: dict, instruction: str | None, done: list, rejected: dict | None) -> str:
    objs, zones = describe_state(sc, sup)
    rules = rule_words(sc, cons)
    head = 'A robot arm rearranges objects on a table. Write the moves that reach the goal.'
    if done or rejected:
        head = ('A robot arm rearranges objects on a table. Some moves are already done; write the REMAINING '
                'moves that reach the goal from the arrangement below.')
    fb = ''
    if done:
        fb += 'Moves already done: ' + '; '.join(f'{x} -> {d}' for x, d in done) + '\n'
    if rejected:
        fb += f'Your move {rejected.get("object")} -> {rejected.get("to")} was rejected: {rejected["reason"]}\n'
    return f"""{head}

Arrangement now:
{objs}

Zones:
{zones}

Task: {instruction or '(given as a goal)'}
Goal: {goal_words(sc, grounded)}
{fb}
One move takes ONE object that has nothing on it and puts it on the table, on top of another object
that has nothing on it, or into a zone with a free slot.
{('Rules: ' + '; '.join(rules)) if rules else ''}
Use only the ids above. Use as few moves as possible. Write the moves in the order they are done.

Return only JSON: {{"moves": [["<object id>", "table" or "<object id>" or "<zone id>"], ...]}}"""


# --------------------------------------------------------------------------- reading the answer
def parse_moves(text: str, sc: dict) -> tuple[list | None, str | None]:
    import openworld
    js, err = openworld._json_answer(text)
    if err:
        return None, err
    raw = js.get('moves', js.get('plan')) if isinstance(js, dict) else js
    if not isinstance(raw, list):
        return None, 'the answer has no "moves" list'
    names = sc['names']
    out = []
    for m in raw:
        if isinstance(m, (list, tuple)) and len(m) >= 2:
            x, d = str(m[0]), str(m[-1])
        elif isinstance(m, dict):
            x = str(m.get('object') or m.get('move') or m.get('obj') or '')
            d = str(m.get('to') or m.get('target') or m.get('on') or m.get('in') or '')
        elif isinstance(m, str) and '->' in m:
            x, d = (s.strip() for s in m.split('->', 1))
        else:
            return None, f'cannot read the move {m!r}'
        x = names.get(x) or x
        d = 'table' if d.lower() in ('table', 'the table') else names.get(d) or d
        out.append((x, d))
    return out, None


# --------------------------------------------------------------------------- the gate
def _key(sup: dict):
    return tuple(sorted(sup.items()))


def gate_moves(sc, sup, moves, cons, goal, visited, gate, optimum, first):
    """Accept moves in order while they pass the gate. -> (accepted, support, rejected or None)."""
    import formula_plan
    import goal_logic
    mover = formula_plan.Mover(sc, cons)
    accepted = []
    for x, dst in moves:
        why = formula_plan.explain(mover, sup, x, dst, first)
        new = {**sup, x: dst}
        if why is None and _key(new) in visited:
            why = 'it returns to an arrangement that was already reached (a loop)'
        if why is None and gate == 'optimal':
            before, after = optimum(sup), optimum(new)
            if before is not None and after is not None and after != before - 1:
                why = (f'it is allowed but not on a shortest plan ({after} moves would remain after it, '
                       f'{before - 1} are possible)')
        if why:
            return accepted, sup, {'object': x, 'to': dst, 'reason': why}
        sup = new
        visited.add(_key(sup))
        accepted.append((x, dst))
        first = None
        if goal.holds(goal_logic.View(sup, sc['items'])):
            break                                         # goal reached: later moves are not needed
    return accepted, sup, None


def _moves_of(steps) -> list:
    out = []
    for st in steps or []:
        a, args = st['action'], st['args']
        if a == 'place-on-table':
            out.append((args[0], 'table'))
        elif a == 'place-on-item':
            out.append((args[0], args[1]))
        elif a == 'place-in':
            out.append((args[0], args[-1]))
    return out


# --------------------------------------------------------------------------- the whole thing
def plan_with_model(sc: dict, grounded, cons: dict, vlm, images=None, instruction: str | None = None,
                    gate: str = 'legal', repairs: int = 1, max_visual_tokens: int = 1024, sample: bool = False) -> dict:
    """-> a plan_for-like dict (status, plan, quality, planner, plan_s, verified, message) plus
    'model_plan' (what the model wrote, what the gate accepted, what the planner added) and 'model_s'."""
    import formula_plan
    import goal_logic
    import openworld
    import plan_quality
    t0 = time.perf_counter()
    gate = gate if gate in GATES else 'legal'
    goal = goal_logic.compile_goal(openworld.goal_formula(grounded), sc)
    first = cons.get('first')
    first = first[0] if isinstance(first, list) and first else first
    ref = openworld.plan_for(sc, grounded, 'auto', cons)              # the planner's own plan: reference + fallback
    ref_moves = (ref.get('quality') or {}).get('moves') if ref['status'] == 'ok' else None
    cache = {}

    def optimum(state: dict):
        k = _key(state)
        if k not in cache:
            p = openworld.plan_for({**sc, 'support': dict(state)}, grounded, 'auto',
                                   {kk: v for kk, v in cons.items() if kk != 'first'})
            q = p.get('quality') or {}
            cache[k] = q.get('moves') if p['status'] == 'ok' and q.get('optimal') else None
        return cache[k]

    sup, visited = dict(sc['support']), {_key(sc['support'])}
    accepted, rejected, texts, calls, model_s = [], [], [], 0, 0.0
    budget = min(4096, max(512, 40 * len(sc['items'])))
    for rnd in range(max(0, repairs) + 1):
        if goal.holds(goal_logic.View(sup, sc['items'])):
            break
        if vlm is None:
            rejected.append({'round': rnd, 'reason': 'no model loaded'})
            break
        p = prompt(sc, sup, grounded, cons, instruction, accepted, rejected[-1] if rejected else None)
        g = vlm.generate(list(images or []), p, max_visual_tokens, max_new_tokens=budget, sample=sample and rnd > 0,
                         context={'mode': 'ow_modelplan', 'scene': {**sc, 'support': dict(sup)}, 'goal': grounded,
                                  'constraints': cons})
        calls += 1
        model_s += g.get('generate_s', 0.0)
        texts.append(g['text'])
        moves, err = parse_moves(g['text'], sc)
        if err:
            rejected.append({'round': rnd, 'reason': err})
            continue
        acc, sup, rej = gate_moves(sc, sup, moves, cons, goal, visited, gate, optimum,
                                   first if not accepted else None)
        accepted += acc
        if rej:
            rejected.append({'round': rnd, 'move': len(accepted) + 1, **rej})
        elif not goal.holds(goal_logic.View(sup, sc['items'])):
            rejected.append({'round': rnd, 'reason': 'the moves end before the goal is reached'})
    # the planner finishes from the arrangement the model reached
    suffix = []
    if not goal.holds(goal_logic.View(sup, sc['items'])):
        rest = openworld.plan_for({**sc, 'support': dict(sup)}, grounded, 'auto',
                                  {kk: v for kk, v in cons.items() if not (kk == 'first' and accepted)})
        if rest['status'] != 'ok':
            return {**rest, 'plan': None, 'planner': 'model+planner',
                    'message': (rest.get('message') or 'no plan') + (
                        f' (after {len(accepted)} model moves)' if accepted else ''),
                    'model_plan': {'gate': gate, 'calls': calls, 'accepted_moves': len(accepted),
                                   'rejected': rejected, 'answers': texts},
                    'model_s': round(model_s, 3), 'plan_s': round(time.perf_counter() - t0 - model_s, 3)}
        suffix = _moves_of(rest['plan'])
    all_moves = accepted + suffix
    acts = formula_plan.to_actions(all_moves, sc)                    # one consistent slot assignment
    if first and acts:
        acts[0] = (acts[0][0].replace('pick-', 'pick-first-', 1), acts[0][1])
    sim, ptext = openworld.verify_acts(sc, acts, cons)
    broken = formula_plan.check_moves(sc, all_moves, cons)
    final = dict(sc['support'])
    for x, dst in all_moves:
        final[x] = dst
    ok = sim['ok'] and broken is None and goal.holds(goal_logic.View(final, sc['items']))
    steps = openworld.robot_plan(acts)
    quality = plan_quality.assess(sc, grounded, steps, 'model', '') if not grounded[0][0] in ('zstack',) else {
        'moves': len(all_moves), 'lower_bound': (ref.get('quality') or {}).get('lower_bound'),
        'optimal': False, 'proof': None, 'max_extra_moves': None, 'moved_more_than_once': []}
    if ref_moves is not None and (ref.get('quality') or {}).get('optimal') and len(all_moves) == ref_moves:
        quality.update(optimal=True, max_extra_moves=0,
                       proof=quality.get('proof') or 'as short as the planner\'s plan, which is proven shortest')
    quality.update(model_moves=len(accepted), planner_moves=len(suffix), reference_moves=ref_moves,
                   extra_moves=None if ref_moves is None else len(all_moves) - ref_moves)
    return {'status': 'ok' if ok else 'error', 'plan': steps, 'quality': quality, 'verified': ok,
            'planner': 'model' if not suffix else 'model+planner' if accepted else 'planner (model moves rejected)',
            'planner_status': 'model', 'problem_pddl': ptext,
            'message': None if ok else 'plan failed verification: ' + (sim['fail_reason'] or broken or 'goal not reached'),
            'model_plan': {'gate': gate, 'calls': calls, 'accepted_moves': len(accepted), 'planner_moves': len(suffix),
                           'rejected': rejected, 'answers': texts},
            'model_s': round(model_s, 3), 'plan_s': round(time.perf_counter() - t0 - model_s, 3)}
