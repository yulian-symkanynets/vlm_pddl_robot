"""Goal formulas: the model writes WHAT must be true at the end, the server finds the moves.

Fixed goal entries ("A on B", "A in the red zone") cover simple tasks. For anything else the
model writes a formula in a small JSON language over a fixed set of facts. The server checks it
(unknown names, wrong types and wrong arity become an exact error message), expands the
quantifiers over the objects of THIS scene, and gives the result to formula_plan.py.

Language (every node has exactly one key, except quantifiers):
    {"and": [F, ...]}  {"or": [F, ...]}  {"not": F}  {"imply": [F, F]}
    {"forall": {"x": "object", "p": "place"}, "where": F (optional), "body": F}
    {"exists": {"x": "object"}, "where": F (optional), "body": F}
Facts about the final arrangement (names are object ids, place ids or variables):
    {"on": [x, y]}         x rests directly on object y
    {"on_table": [x]}      x rests on the table
    {"in": [x, p]}         x is in place p (zone, box, tray, ...)
    {"clear": [x]}         nothing rests on x
    {"above": [x, y]}      x is somewhere above y in the same stack
Facts that do not change (decided while expanding):
    {"color": [x, "red"]}  {"shape": [x, "cube"]}  {"same_color": [x, y]}
    {"same_shape": [x, y]} {"is": [x, y]}          (x and y are the same object / place)
"""
from __future__ import annotations

import itertools

STATE_ATOMS = {'on': 2, 'on_table': 1, 'in': 2, 'clear': 1, 'above': 2}
STATIC_ATOMS = {'color': 2, 'shape': 2, 'same_color': 2, 'same_shape': 2, 'is': 2}
KINDS = {'object': 'object', 'item': 'object', 'place': 'place', 'zone': 'place'}
MAX_NODES = 150000               # a formula that expands to more than this is refused


class FormulaError(ValueError):
    pass


def stack_in_zones_formula(zone_of: dict) -> dict:
    """One tower per colour standing in the zone of that colour: every object of colour c is directly
    in zone c or on an object of colour c, and only one object of colour c stands directly in zone c.
    (The chain under an object of colour c can then only end in zone c: one tower, base in the zone.)"""
    parts = []
    for c, z in sorted(zone_of.items()):
        parts.append({'forall': {'x': 'object'}, 'where': {'color': ['x', c]}, 'body': {'or': [
            {'in': ['x', z]},
            {'exists': {'y': 'object'}, 'where': {'color': ['y', c]}, 'body': {'on': ['x', 'y']}}]}})
        parts.append({'forall': {'x': 'object', 'y': 'object'},
                      'where': {'and': [{'color': ['x', c]}, {'color': ['y', c]}, {'not': {'is': ['x', 'y']}}]},
                      'body': {'not': {'and': [{'in': ['x', z]}, {'in': ['y', z]}]}}})
    return {'and': parts}


# "every stack one colour" as a formula: the general path for it when the exact search
# (colour_stacks.py) is too slow, and an example of the language.
STACK_BY_COLOUR = {"and": [
    {"forall": {"x": "object"}, "body": {"or": [
        {"on_table": ["x"]},
        {"exists": {"y": "object"}, "where": {"same_color": ["x", "y"]}, "body": {"on": ["x", "y"]}}]}},
    {"forall": {"x": "object", "y": "object"},
     "where": {"and": [{"same_color": ["x", "y"]}, {"not": {"is": ["x", "y"]}}]},
     "body": {"not": {"and": [{"on_table": ["x"]}, {"on_table": ["y"]}]}}}]}


# --------------------------------------------------------------------------- small boolean algebra
def _and(parts):
    out = []
    for p in parts:
        if p is True:
            continue
        if p is False:
            return False
        out.extend(p[1] if isinstance(p, tuple) and p[0] == 'and' else [p])
    return True if not out else out[0] if len(out) == 1 else ('and', out)


def _or(parts):
    out = []
    for p in parts:
        if p is False:
            continue
        if p is True:
            return True
        out.extend(p[1] if isinstance(p, tuple) and p[0] == 'or' else [p])
    return False if not out else out[0] if len(out) == 1 else ('or', out)


def _not(p):
    if isinstance(p, bool):
        return not p
    return p[1] if p[0] == 'not' else ('not', p)


# --------------------------------------------------------------------------- expansion
class Goal:
    """A formula expanded for one scene. `conjuncts` are the requirements that must all hold."""

    def __init__(self, tree, formula):
        self.tree, self.formula = tree, formula
        self.conjuncts = tree[1] if isinstance(tree, tuple) and tree[0] == 'and' else [tree]

    def violated(self, view: 'View') -> int:
        return sum(1 for c in self.conjuncts if not holds(c, view))

    def holds(self, view: 'View') -> bool:
        return all(holds(c, view) for c in self.conjuncts)


def compile_goal(formula, sc: dict) -> Goal:
    """Check `formula` against the scene and expand its quantifiers."""
    if isinstance(formula, str):
        import json
        try:
            formula = json.loads(formula)
        except json.JSONDecodeError as e:
            raise FormulaError(f'the formula is not valid JSON: {e}')
    items, places = sc['items'], sc['env']['places']
    count = [0]

    def term(a, env, kind, where):
        if not isinstance(a, str):
            raise FormulaError(f'{where}: names must be text, got {a!r}')
        if a not in env and a not in items and a not in places:
            raise FormulaError(f'{where}: "{a}" is not an object, a place or a variable of this formula '
                               f'(objects: {", ".join(list(items)[:6])}...)')
        t = env.get(a, a)
        if kind == 'object' and t not in items:
            raise FormulaError(f'{where}: "{a}" must be an object, but it is a place')
        if kind == 'place' and t not in places:
            raise FormulaError(f'{where}: "{a}" must be a place, but it is an object')
        return t

    def attr(x, key):
        return str(items[x].get(key) or '').lower() or None

    def ground(node, env):
        count[0] += 1
        if count[0] > MAX_NODES:
            raise FormulaError('the formula is too large for this scene (too many nested "forall"/"exists")')
        if not isinstance(node, dict) or not node:
            raise FormulaError(f'every part of the formula must be an object like {{"on": ["a", "b"]}}, got {node!r}')
        q = 'forall' if 'forall' in node else 'exists' if 'exists' in node else None
        if q:
            spec = node[q]
            extra = set(node) - {q, 'where', 'body'}
            if extra or 'body' not in node or not isinstance(spec, dict) or not spec:
                raise FormulaError(f'"{q}" needs {{"{q}": {{"x": "object"}}, "where": optional, "body": ...}}')
            domains, names = [], list(spec)
            for v, kind in spec.items():
                if v in items or v in places:
                    raise FormulaError(f'variable "{v}" has the same name as an object or place')
                if KINDS.get(str(kind)) is None:
                    raise FormulaError(f'variable "{v}" has the unknown type "{kind}" (use "object" or "place")')
                domains.append(list(items) if KINDS[kind] == 'object' else list(places))
            parts = []
            for combo in itertools.product(*domains):
                e = {**env, **dict(zip(names, combo))}
                body = ground(node['body'], e)
                if 'where' in node:
                    w = ground(node['where'], e)
                    parts.append(_or([_not(w), body]) if q == 'forall' else _and([w, body]))
                else:
                    parts.append(body)
            return _and(parts) if q == 'forall' else _or(parts)
        if len(node) != 1:
            raise FormulaError(f'a part of the formula has several operators {sorted(node)}; '
                               'use one operator per object')
        (op, args), = node.items()
        if op in ('and', 'or'):
            if not isinstance(args, list) or not args:
                raise FormulaError(f'"{op}" needs a non-empty list')
            parts = [ground(a, env) for a in args]
            return _and(parts) if op == 'and' else _or(parts)
        if op == 'not':
            return _not(ground(args, env))
        if op == 'imply':
            if not isinstance(args, list) or len(args) != 2:
                raise FormulaError('"imply" needs a list of exactly two parts')
            return _or([_not(ground(args[0], env)), ground(args[1], env)])
        arity = STATE_ATOMS.get(op) or STATIC_ATOMS.get(op)
        if arity is None:
            raise FormulaError(f'unknown operator "{op}" (allowed: and, or, not, imply, forall, exists, '
                               + ', '.join(list(STATE_ATOMS) + list(STATIC_ATOMS)) + ')')
        if not isinstance(args, list) or len(args) != arity:
            raise FormulaError(f'"{op}" needs a list of {arity} name(s)')
        w = f'"{op}"'
        if op in ('color', 'shape'):
            x = term(args[0], env, 'object', w)
            return attr(x, op) == str(args[1]).lower()
        if op in ('same_color', 'same_shape'):
            x, y = term(args[0], env, 'object', w), term(args[1], env, 'object', w)
            key = op.split('_')[1]
            return attr(x, key) is not None and attr(x, key) == attr(y, key)
        if op == 'is':
            x = term(args[0], env, None, w)
            y = term(args[1], env, None, w)
            return x == y
        kinds = {'on': ('object', 'object'), 'on_table': ('object',), 'in': ('object', 'place'),
                 'clear': ('object',), 'above': ('object', 'object')}[op]
        ts = tuple(term(a, env, k, w) for a, k in zip(args, kinds))
        if op in ('on', 'above') and ts[0] == ts[1]:
            return False                                   # nothing rests on itself
        return ('atom', op, ts)

    tree = ground(formula, {})
    if tree is False:
        raise FormulaError('this goal can never be true for these objects (a fixed fact such as a colour '
                           'or shape makes it impossible)')
    return Goal(tree, formula)


# --------------------------------------------------------------------------- evaluation on a state
class View:
    """The arrangement: support[x] is "table", an object id or a place id."""

    def __init__(self, support: dict, items):
        self.sup, self.items, self._covered = support, items, None

    @property
    def covered(self) -> set:
        if self._covered is None:
            self._covered = set(self.sup.values())
        return self._covered


def holds(g, v: View) -> bool:
    if g is True or g is False:
        return g
    tag = g[0]
    if tag == 'and':
        return all(holds(p, v) for p in g[1])
    if tag == 'or':
        return any(holds(p, v) for p in g[1])
    if tag == 'not':
        return not holds(g[1], v)
    _, op, a = g
    if op == 'on':
        return v.sup.get(a[0]) == a[1]
    if op == 'on_table':
        return v.sup.get(a[0]) == 'table'
    if op == 'in':
        return v.sup.get(a[0]) == a[1]
    if op == 'clear':
        return a[0] not in v.covered
    cur = v.sup.get(a[0])                                  # above
    while cur in v.items:
        if cur == a[1]:
            return True
        cur = v.sup.get(cur)
    return False


# --------------------------------------------------------------------------- plain words
def describe(formula, sc: dict | None = None) -> str:
    """The formula in plain English, built from the formula itself, so it shows what the server
    will really aim for, not what the model says it meant."""
    def w(n) -> str:
        if not isinstance(n, dict) or not n:
            return '?'
        q = 'forall' if 'forall' in n else 'exists' if 'exists' in n else None
        if q:
            vs = n[q] if isinstance(n[q], dict) else {}
            vars_ = ' and '.join(f'{v} ({KINDS.get(str(k), k)})' for v, k in vs.items())
            where = f' where {w(n["where"])}' if 'where' in n else ''
            head = 'for every' if q == 'forall' else 'there is some'
            tail = ': ' if q == 'forall' else ' such that '
            return f'{head} {vars_}{where}{tail}{w(n.get("body"))}'
        (op, a), = n.items() if len(n) == 1 else [('?', None)]
        if op in ('and', 'or') and isinstance(a, list):
            return '(' + (f' {op} ').join(w(p) for p in a) + ')'
        if op == 'not':
            return f'not ({w(a)})'
        if op == 'imply' and isinstance(a, list) and len(a) == 2:
            return f'if {w(a[0])} then {w(a[1])}'
        if not isinstance(a, list):
            return '?'
        t = {'on': '{0} is directly on {1}', 'on_table': '{0} is on the table', 'in': '{0} is in {1}',
             'clear': 'nothing is on {0}', 'above': '{0} is somewhere above {1}',
             'color': '{0} is {1}', 'shape': '{0} is a {1}', 'same_color': '{0} and {1} have the same colour',
             'same_shape': '{0} and {1} have the same shape', 'is': '{0} is {1}'}.get(op)
        return t.format(*[str(x) for x in a]) if t and len(a) >= t.count('{') else f'{op}{a}'
    return w(formula)
