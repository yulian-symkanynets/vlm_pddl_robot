"""How good is a plan? A proof, not a promise.

A plan returned by the server is always valid (it is simulated before it is sent).
Whether it is also the SHORTEST plan is a separate question, answered here with a
lower bound that needs no search:

    lower_bound = number of objects that must be moved at least once

An object must move if its own goal fact is not true yet, if another object has to be
placed where it stands, or if it stands on top of an object that must move (it has to
get out of the way first). Every such object costs at least one pick and one place, so
no plan can have fewer moves than this. Then:

    moves == lower_bound          the plan is proven shortest, no search needed
    planner was optimal A*        the plan is proven shortest by the planner
    otherwise                     the plan has at most (moves - lower_bound) extra moves

A "wasted move" in the sense of "put blue on red, then move blue again" is detected
directly: an object picked up more than once in the plan. That is not always wrong
(parking an object in a temporary place can be necessary), but in a plan proven
shortest every repeated move is unavoidable.
"""
from __future__ import annotations

OPTIMAL_PLANNERS = ("fast-downward-opt",)


def must_move(support: dict[str, str], items, goal: list[tuple[str, ...]]) -> set[str]:
    """Objects that every valid plan has to pick up at least once."""
    above = {y: x for x, y in support.items() if y in items}
    need: set[str] = set()
    for g in goal:
        x = g[1]
        if x not in support:                        # unknown support: nothing can be said
            continue
        if g[0] == "on-table":
            if support[x] != "table":
                need.add(x)
        elif g[0] in ("on", "in"):
            y = g[2]
            if support[x] != y:
                need.add(x)
                if y in above and above[y] != x:    # something else sits where x must go
                    need.add(above[y])
    for x in list(need):                            # whatever stands on a mover moves too
        while x in above:
            x = above[x]
            need.add(x)
    return need


def lower_bound(support: dict[str, str], items, goal: list[tuple[str, ...]]) -> int:
    """Fewest moves any plan needs. For fixed goals: the objects that must move. For the goal
    "one stack per colour" ((col-ok x) for every object):
      - an object on an object of another colour, or in a place, must move, and so must
        everything stacked on it;
      - of the objects of one colour that stand on the table and are not counted yet, at most
        one can stay as the base, so all but one of them must move."""
    if not any(g[0] == "col-ok" for g in goal):
        return len(must_move(support, items, goal))
    colour = {x: (items[x].get("color") if isinstance(items[x], dict) else None) for x in items}
    above = {y: x for x, y in support.items() if y in items}
    need = set()
    for x, y in support.items():
        if x not in colour or colour[x] is None:
            continue
        if (y in items and colour.get(y) != colour[x]) or (y != "table" and y not in items):
            need.add(x)
    for x in list(need):
        while x in above:
            x = above[x]
            need.add(x)
    on_table = {}
    for x, y in support.items():
        if y == "table" and x not in need and colour.get(x):
            on_table[colour[x]] = on_table.get(colour[x], 0) + 1
    return len(need) + sum(max(0, k - 1) for k in on_table.values())


def moves_of(plan: list[dict] | list[tuple]) -> list[str]:
    """Objects picked up, in order, from a plan in the server's format or as (name, args)."""
    out = []
    for step in plan or []:
        name, args = (step["action"], step["args"]) if isinstance(step, dict) else step
        if name.startswith("pick"):
            out.append(args[0])
    return out


def assess(sc: dict, goal, plan, planner: str, planner_status: str = "") -> dict:
    """Quality certificate for a verified plan on the normalized scene `sc`."""
    picked = moves_of(plan)
    if any(g[0] == "formula" for g in goal):         # general goal: verified, but no lower bound
        return {"moves": len(picked), "lower_bound": None, "optimal": len(picked) == 0,
                "proof": "the goal already holds" if not picked else None, "max_extra_moves": None,
                "moved_more_than_once": sorted({x for x in picked if picked.count(x) > 1}),
                "note": None if not picked else "general goal: the plan is verified, not proven shortest"}
    lb = lower_bound(sc["support"], sc["items"], goal)
    moved_twice = sorted({x for x in picked if picked.count(x) > 1})
    if planner in OPTIMAL_PLANNERS and "solved" in planner_status.lower():
        proof = "optimal search: Fast Downward astar(lmcut()), an admissible heuristic"
    elif len(picked) == lb:
        proof = "the plan moves only objects that must move, each once"
    else:
        proof = None
    out = {
        "moves": len(picked),
        "lower_bound": lb,
        "optimal": proof is not None,
        "proof": proof,
        "max_extra_moves": 0 if proof else len(picked) - lb,
        "moved_more_than_once": moved_twice,
    }
    if all(g[0] in ("on", "in", "on-table") for g in goal):
        out["must_move"] = sorted(must_move(sc["support"], sc["items"], goal))   # what the lower bound counts
    return out
