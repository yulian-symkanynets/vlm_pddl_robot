"""Task-specific PDDL: the allowed moves change with the task.

domain/tabletop.pddl describes what the robot can do in general. A task may forbid some
of it ("do not put anything on the table", "leave the purple cylinder where it is",
"never put anything on X", "first move the green block"), and some objects cannot carry
others (a prism). For every task this module

  * builds a DOMAIN from tabletop.pddl with only the actions and preconditions the task
    needs (no constraint -> exactly tabletop.pddl, so nothing else changes), and
  * returns the extra :init FACTS that say what is permitted.

    constraint                         domain change                         problem facts
    no_table (or never_on *, table)    action place-on-table removed         -
    never_on [x, "table"]              place-on-table needs (can-table ?x)   can-table for others
    keep [x, ...]                      every pick needs (movable ?x)         movable for others
    never_on [x|*, y] / y has no top   place-on-item needs (can-stack ?x ?y) can-stack for allowed pairs
    never_on [x|*, place]              -                                     (fits x place) removed
    first [x]                          picks need (unlocked); pick-first-*   (first-object x)
                                       actions pick x and add (unlocked)
    max_stack_height                   not expressible here: reported, use the task server

Plain STRIPS + typing, so Fast Downward and pyperplan still work.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

from scenes import DOMAIN_PATH

NO_FLAT_TOP = ("prism", "pyramid", "cone", "ball", "sphere", "triangle", "wedge")
PICKS = ("pick-from-table", "pick-from-item", "pick-from-place")


def normalize_constraints(raw, sc: dict) -> tuple[dict, list[str]]:
    """Ground constraint ids on the scene. Returns (constraints, problems for the user)."""
    if not raw:
        return {}, []
    names, items, places = sc["names"], sc["items"], sc["env"]["places"]
    out, problems = {}, []

    def obj(x):
        if x in ("*", "table"):
            return x
        y = names.get(x)
        if y is None:
            problems.append(f"constraint names {x!r}, which is not in the scene")
        return y

    if raw.get("no_table") or raw.get("no_temporary"):
        out["no_table"] = True
    keep = [obj(x) for x in raw.get("keep", [])]
    if keep:
        out["keep"] = [k for k in keep if k in items]
    never = []
    for pair in raw.get("never_on", []):
        a, t = (pair.get("object"), pair.get("on")) if isinstance(pair, dict) else pair
        a, t = obj(a), obj(t)
        if a is not None and t is not None:
            never.append((a, t))
    if never:
        out["never_on"] = never
    first = raw.get("first") or []
    first = [first] if isinstance(first, str) else list(first)
    first = [obj(x) for x in first]
    if first and first[0] in items:
        out["first"] = first[0]
        if len(first) > 1:
            problems.append("only the first object of 'first' can be enforced in the PDDL path")
    if raw.get("max_stack_height"):
        out["max_stack_height"] = raw["max_stack_height"]
    for a, t in never:
        if t not in items and t not in places and t != "table":
            problems.append(f"never_on target {t!r} is unknown")
    return out, [p for p in problems if not p.startswith("only")]


# ---------------------------------------------------------------- domain text surgery

def _blocks(text: str) -> dict[str, tuple[int, int]]:
    """name -> (start, end) of every (:action ...) block."""
    out = {}
    for m in re.finditer(r"\(:action\s+([\w-]+)", text):
        depth, i = 0, m.start()
        for j in range(i, len(text)):
            depth += text[j] == "("
            depth -= text[j] == ")"
            if depth == 0:
                out[m.group(1)] = (i, j + 1)
                break
    return out


def _add_pre(block: str, fact: str) -> str:
    return block.replace(":precondition (and ", f":precondition (and {fact} ", 1) \
        if ":precondition (and " in block else \
        re.sub(r":precondition \(([^()]*\([^()]*\))\)", rf":precondition (and \1 {fact})", block, count=1)


def make_domain(cons: dict, sc: dict, base: str | None = None) -> str:
    text = base if base is not None else DOMAIN_PATH.read_text()
    items = sc["items"]
    needs_stack = bool(cons.get("never_on")) and any(t in items for _, t in cons["never_on"]) \
        or any(_no_top(d) for d in items.values())
    table_ban_all = cons.get("no_table") or any(a == "*" and t == "table" for a, t in cons.get("never_on", []))
    table_ban_some = not table_ban_all and any(t == "table" for _, t in cons.get("never_on", []))
    preds = []
    b = _blocks(text)

    def edit(name, fn):
        nonlocal text, b
        s, e = b[name]
        text = text[:s] + fn(text[s:e]) + text[e:]
        b = _blocks(text)

    if cons.get("keep"):
        preds.append("(movable ?x - item)")
        for p in PICKS:
            edit(p, lambda blk: _add_pre(blk, "(movable ?x)"))
    if needs_stack:
        preds.append("(can-stack ?x - item ?y - item)")
        edit("place-on-item", lambda blk: _add_pre(blk, "(can-stack ?x ?y)"))
    if table_ban_some:
        preds.append("(can-table ?x - item)")
        edit("place-on-table", lambda blk: _add_pre(blk, "(can-table ?x)"))
    if cons.get("first"):
        preds += ["(unlocked)", "(first-object ?x - item)"]
        firsts = []
        for p in PICKS:
            s, e = b[p]
            orig = text[s:e]
            f = orig.replace(f"(:action {p}", f"(:action {p.replace('pick-', 'pick-first-')}", 1)
            f = _add_pre(f, "(first-object ?x)")
            f = f.replace(":effect (and ", ":effect (and (unlocked) (not (first-object ?x)) ", 1)
            firsts.append(f)
            edit(p, lambda blk: _add_pre(blk, "(unlocked)"))
        s, e = b["place-on-table"] if "place-on-table" in b else b[PICKS[-1]]
        text = text[:e] + "\n\n  " + "\n\n  ".join(firsts) + text[e:]
        b = _blocks(text)
    if table_ban_all and "place-on-table" in b:
        s, e = b["place-on-table"]
        text = text[:s] + "; place-on-table removed: this task forbids putting objects on the table" + text[e:]
    if preds:
        text = re.sub(r"(\(:predicates)", r"\1\n    " + "\n    ".join(preds), text, count=1)
    return text


def _no_top(d: dict) -> bool:
    label = (d.get("label") or "").lower()
    return d.get("can_support") is False or any(w in label for w in NO_FLAT_TOP)


def constraint_facts(cons: dict, sc: dict) -> list[str]:
    items = sc["items"]
    never = cons.get("never_on", [])
    facts = []
    if cons.get("keep"):
        facts += [f"(movable {x})" for x in items if x not in cons["keep"]]
    needs_stack = bool(never) and any(t in items for _, t in never) or any(_no_top(d) for d in items.values())
    if needs_stack:
        for x in items:
            for y in items:
                if x != y and not _no_top(items[y]) and not any(a in (x, "*") and t == y for a, t in never):
                    facts.append(f"(can-stack {x} {y})")
    if not cons.get("no_table") and any(t == "table" for _, t in never) \
            and not any(a == "*" and t == "table" for a, t in never):
        facts += [f"(can-table {x})" for x in items if not any(a == x and t == "table" for a, t in never)]
    if cons.get("first"):
        facts.append(f"(first-object {cons['first']})")
    elif cons:
        pass
    return facts


def apply_to_problem(ptext: str, cons: dict, sc: dict) -> str:
    """Add the permission facts to :init and drop (fits x place) for forbidden places."""
    places = sc["env"]["places"]
    for a, t in cons.get("never_on", []):
        if t in places:
            xs = list(sc["items"]) if a == "*" else [a]
            for x in xs:
                ptext = ptext.replace(f"(fits {x} {t})", "")
    facts = constraint_facts(cons, sc)
    if not facts:
        return ptext
    return ptext.replace("(:init\n    ", "(:init\n    " + "\n    ".join(facts) + "\n    ", 1)


def needs_custom_domain(cons: dict, sc: dict) -> bool:
    return bool(cons.get("keep") or cons.get("never_on") or cons.get("no_table") or cons.get("first")
                or any(_no_top(d) for d in sc["items"].values()))


def write_domain(text: str) -> Path:
    f = tempfile.NamedTemporaryFile("w", suffix=".pddl", prefix="domain_", delete=False)
    f.write(text)
    f.close()
    return Path(f.name)
