"""Task specification: what the robot is asked to achieve, beyond colour sorting.

A task has goals (facts that must hold at the end) and constraints (rules every
placement must respect). Wire format, inside the scene message or on its own:

    "task": {
      "goal": [
        {"sort_by_color": true},                       every object into the zone of its colour
        {"object": "a", "in": "red_zone"},             a ends in a stack standing in red_zone
        {"object": "a", "on": "b"},                    a rests directly on b
        {"object": "a", "on": "table"},                a stands alone on a free slot / the table
        {"object": "a", "clear": true},                nothing on a
        {"stack": ["a", "b", "c"], "in": "blue_zone"}  tower, listed bottom to top; "in" optional
      ],
      "constraints": {
        "no_temporary": true,                          do not park objects in free slots
        "keep": ["x"],                                 never move these objects
        "never_on": [["a", "b"], ["*", "tmp_1"]],      never place a on b; nothing on tmp_1
        "max_stack_height": 2
      }
    }

In "never_on" the second entry may be an object id, a location id, or a kind:
"slot" (any free slot), "zone" (any zone floor), "object" (any object).
"""
from __future__ import annotations

from dataclasses import dataclass, field


class SceneError(ValueError):
    """The semantic scene is structurally invalid (not merely uncertain)."""


class TaskError(SceneError):
    """The task is malformed or contradicts itself (reported like a bad scene, HTTP 422)."""


KINDS = ("slot", "zone", "object")


@dataclass(frozen=True)
class Task:
    atoms: tuple = ()                    # ("in", o, zone) ("on", o, support) ("on_table", o) ("clear", o)
                                         # ("not_on", o, support)
    sort: bool = False                   # every remaining object into the zone of its colour
    never_on: tuple = ()                 # (object or "*", target)
    keep: frozenset = frozenset()
    no_temporary: bool = False
    max_height: int | None = None
    raw: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def constrained(self) -> bool:
        return bool(self.never_on or self.keep or self.no_temporary or self.max_height)

    @property
    def simple(self) -> bool:
        """Only zone goals and no constraints: the fast sorting planner can handle it."""
        return not self.constrained and all(a[0] == "in" for a in self.atoms)

    # ------------------------------------------------------------ goals
    def zone_goals(self, scene) -> dict[str, str]:
        """object -> zone for the 'in' goals, plus colour sorting for the rest if asked."""
        out = {}
        if self.sort:
            placed = {a[1] for a in self.atoms}
            for o in scene.objects:
                if o not in placed and o not in self.keep and scene.trusted(o) \
                        and (z := scene.zone_of(o)) is not None:
                    out[o] = z
        out.update({a[1]: a[2] for a in self.atoms if a[0] == "in"})
        return out

    def unmet(self, scene) -> list[str]:
        """Goal facts that do not hold in the scene, as readable strings."""
        out = []
        frozen = scene.frozen()
        for o, z in sorted(self.zone_goals(scene).items()):
            if scene.base(o) != z and not (self.sort and o in frozen and ("in", o, z) not in self.atoms):
                out.append(f"{o} in {z}")
        clear = scene.clear()
        for a in self.atoms:
            if a[0] == "on" and scene.on[a[1]] != a[2]:
                out.append(f"{a[1]} on {a[2]}")
            elif a[0] == "on_table" and not _on_table(scene, a[1]):
                out.append(f"{a[1]} on the table")
            elif a[0] == "clear" and a[1] not in clear:
                out.append(f"{a[1]} clear")
            elif a[0] == "not_on" and scene.on[a[1]] == a[2]:
                out.append(f"{a[1]} off {a[2]}")
        return out

    def satisfied(self, scene) -> bool:
        return not self.unmet(scene)

    def must_move(self, scene) -> set[str]:
        """Objects that certainly have to be moved at least once (admissible heuristic)."""
        up = scene.above()
        need: set[str] = set()
        for o, z in self.zone_goals(scene).items():
            if scene.base(o) != z:
                need.add(o)
        for a in self.atoms:
            if a[0] == "on" and scene.on[a[1]] != a[2]:
                need.add(a[1])
                if a[2] in up:                       # the support must be uncovered first
                    need.add(up[a[2]])
            elif a[0] == "on_table" and not _on_table(scene, a[1]):
                need.add(a[1])
            elif a[0] == "clear" and a[1] in up:
                need.add(up[a[1]])
            elif a[0] == "not_on" and scene.on[a[1]] == a[2]:
                need.add(a[1])
        for o in list(need):                         # everything above a mover moves too
            while o in up:
                o = up[o]
                need.add(o)
        return need

    # ------------------------------------------------------------ constraints
    def violation(self, scene, obj: str, dst: str) -> str | None:
        """Reason this placement breaks a constraint, or None."""
        is_obj = dst in scene.objects
        kind = "object" if is_obj else {"buffer": "slot", "zone": "zone"}.get(scene.locs[dst].kind, "table")
        region = scene.base(dst) if is_obj else dst
        if self.no_temporary and kind == "slot" and ("on_table", obj) not in self.atoms \
                and ("on", obj, dst) not in self.atoms:
            return "the task forbids temporary slots"
        for a, t in self.never_on:
            if a in (obj, "*") and t in (dst, kind, region):
                return f"the task forbids placing {obj} on {t}"
        if self.max_height is not None:
            height, x = 1, dst
            while x in scene.objects:
                height, x = height + 1, scene.on[x]
            if height > self.max_height:
                return f"the task limits stacks to {self.max_height} object(s)"
        return None

    def describe(self) -> str:
        """Plain-language statement of the task (for prompts and logs)."""
        parts = []
        if self.sort:
            parts.append("every object must end in the zone of its own colour")
        for a in self.atoms:
            parts.append({"in": lambda: f"{a[1]} must end in {a[2]}",
                          "on": lambda: f"{a[1]} must rest directly on {a[2]}",
                          "on_table": lambda: f"{a[1]} must stand alone on a free slot",
                          "clear": lambda: f"nothing may be on {a[1]}",
                          "not_on": lambda: f"{a[1]} must not rest on {a[2]}"}[a[0]]())
        if self.no_temporary:
            parts.append("free slots must not be used for parking")
        for a, t in self.never_on:
            parts.append(f"never place {'anything' if a == '*' else a} on {t}")
        if self.keep:
            parts.append("do not move " + ", ".join(sorted(self.keep)))
        if self.max_height:
            parts.append(f"no stack higher than {self.max_height}")
        return "; ".join(parts) or "nothing to do"


def _on_table(scene, o: str) -> bool:
    s = scene.on[o]
    return s in scene.locs and scene.locs[s].kind in ("buffer", "table")


def parse_task(d, objects: dict, locs: dict) -> Task | None:
    """Validate a task message against the scene. Raises TaskError with a readable reason."""
    if d is None:
        return None
    if not isinstance(d, dict):
        raise TaskError("task must be an object with 'goal' and optional 'constraints'")
    goal = d.get("goal", [])
    if isinstance(goal, dict):
        goal = [goal]
    if not isinstance(goal, list):
        raise TaskError("'goal' must be a list")
    zones = {l for l, v in locs.items() if v.kind == "zone"}
    slots = {l for l, v in locs.items() if v.kind == "buffer"}

    def obj(x):
        if x not in objects:
            raise TaskError(f"unknown object {x!r}")
        return x

    atoms, sort = [], bool(d.get("sort_by_color"))
    for g in goal:
        if g == "sort_by_color" or (isinstance(g, dict) and g.get("sort_by_color")):
            sort = True
            continue
        if not isinstance(g, dict):
            raise TaskError(f"goal entry {g!r} is not understood")
        if "stack" in g:
            chain = [obj(x) for x in g["stack"]]
            if len(set(chain)) != len(chain) or not chain:
                raise TaskError("a stack must list different objects, bottom to top")
            base = g.get("in", g.get("on"))
            if base is not None:
                g0 = {"object": chain[0], ("in" if base in zones else "on"): base}
                goal.append(g0)
            atoms += [("on", top, below) for below, top in zip(chain, chain[1:])]
            continue
        o = obj(g.get("object"))
        if g.get("clear"):
            atoms.append(("clear", o))
        off = g.get("not_on", g.get("off"))
        if off is not None:
            if off not in objects and off not in locs:
                raise TaskError(f"unknown support {off!r} in not_on")
            atoms.append(("not_on", o, off))
            if g.get("in", g.get("on")) is None:
                continue
        target = g.get("in", g.get("on"))
        if target is None:
            if not g.get("clear"):
                raise TaskError(f"goal for {o} needs 'on', 'in', 'not_on' or 'clear'")
            continue
        if target == "table":
            atoms.append(("on_table", o))
        elif target in zones:
            atoms.append(("in", o, target))
        elif target in slots or target in objects:
            atoms.append(("on", o, target))
        else:
            raise TaskError(f"unknown destination {target!r} for {o}")
    atoms = list(dict.fromkeys(atoms))
    # consistency
    for a in atoms:
        if a[0] == "on" and a[1] == a[2]:
            raise TaskError(f"{a[1]} cannot rest on itself")
        if a[0] == "on" and a[2] in objects and not objects[a[2]].can_support:
            raise TaskError(f"nothing can be placed on {a[2]} ({objects[a[2]].type})")
    below = [a[2] for a in atoms if a[0] == "on" and a[2] in objects]
    dup = sorted({b for b in below if below.count(b) > 1})
    if dup:
        raise TaskError(f"more than one object should rest on {', '.join(dup)}")
    direct = [a[1] for a in atoms if a[0] in ("on", "on_table")]
    if len(direct) != len(set(direct)):
        raise TaskError("an object has two different supports in the goal")
    zone_of = {}
    for a in atoms:
        if a[0] == "in" and zone_of.setdefault(a[1], a[2]) != a[2]:
            raise TaskError(f"{a[1]} has two different goal zones")
    on = {a[1]: a[2] for a in atoms if a[0] == "on"}
    for x in on:
        seen, cur = {x}, on[x]
        while cur in on:
            if cur in seen:
                raise TaskError(f"circular stacking through {x}")
            seen.add(cur)
            cur = on[cur]
    for a in atoms:
        if a[0] == "not_on" and ("on", a[1], a[2]) in atoms:
            raise TaskError(f"{a[1]} should rest on {a[2]} and not rest on it")
    covered = {a[2] for a in atoms if a[0] == "on"}
    for a in atoms:
        if a[0] == "clear" and a[1] in covered:
            raise TaskError(f"{a[1]} should be clear and carry another object at the same time")
    for x, z in zone_of.items():                     # a stack is in one zone only
        cur = x
        while cur in on:
            cur = on[cur]
            if cur in zone_of and zone_of[cur] != z:
                raise TaskError(f"{x} should be in {z} but rests on {cur}, which should be in {zone_of[cur]}")
    c = d.get("constraints") or {}
    if not isinstance(c, dict):
        raise TaskError("'constraints' must be an object")
    never = []
    for pair in c.get("never_on", []):
        a, t = (pair["object"], pair["on"]) if isinstance(pair, dict) else pair
        if a != "*":
            obj(a)
        if t not in objects and t not in locs and t not in KINDS:
            raise TaskError(f"unknown target {t!r} in never_on")
        never.append((a, t))
    keep = frozenset(obj(x) for x in c.get("keep", []))
    mh = c.get("max_stack_height")
    if mh is not None and (not isinstance(mh, int) or mh < 1):
        raise TaskError("max_stack_height must be a positive integer")
    for a, t in never:
        for g in atoms:
            if g[0] == "on" and a in (g[1], "*") and t in (g[2], "object" if g[2] in objects else "slot"):
                raise TaskError(f"the goal '{g[1]} on {g[2]}' contradicts never_on")
    task = Task(tuple(atoms), sort, tuple(never), keep, bool(c.get("no_temporary")), mh, raw=d)
    if not task.atoms and not task.sort:
        raise TaskError("the task has no goal")
    return task
