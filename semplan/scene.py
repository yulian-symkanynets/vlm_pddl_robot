"""Semantic scene model, action model and plan validator.

The whole planner works on one small symbolic state:

    on[obj] = support        support is another object id, a location id, or HELD

Locations are the only places an object can rest besides another object:

    zone    a colour target area with a floor capacity (number of ground poses)
    buffer  a verified free temporary pose (capacity 1), supplied by the laptop
    table   wherever objects happen to stand at the start; never a place target

The region of an object is the location at the bottom of its stack, so an object
stacked on something inside the red zone is itself "in the red zone".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .task import SceneError, Task, TaskError, parse_task  # noqa: F401

HELD = "__gripper__"
TABLE = "table"

# Shapes with a flat top. The laptop can override this per object ("can_support").
FLAT_TOP = {"cube", "cylinder", "box", "cuboid"}




@dataclass(frozen=True)
class Obj:
    id: str
    type: str = "cube"
    color: str | None = None        # None = not identified
    can_support: bool = True        # may another object be placed on top?
    confidence: float = 1.0         # identity confidence (class + colour)
    pose_confidence: float = 1.0    # how sure the laptop is about pose / support
    reachable: bool = True          # local feasibility flag from the robot side
    position: tuple | None = None   # metric pose, carried through, never planned on


@dataclass(frozen=True)
class Loc:
    id: str
    kind: str                       # "zone" | "buffer" | "table"
    color: str | None = None        # zones only
    capacity: int = 1               # ground poses available in this location


@dataclass
class Scene:
    objects: dict[str, Obj]
    locs: dict[str, Loc]
    on: dict[str, str]                               # object id -> support
    version: int = 0
    scene_id: str = "scene"
    min_confidence: float = 0.8
    occluded: tuple[str, ...] = ()                   # objects with a hidden region near them
    explicit_goals: dict[str, str] | None = None     # object -> zone; None = sort by colour
    task: Task | None = None                         # general task (goals + constraints)
    # Collision risk reported by the laptop: target -> ((blocker, where the blocker stood), ...).
    # The target may not be picked while a blocker still stands where it was reported.
    blockers: dict = field(default_factory=dict)

    # ------------------------------------------------------------ derived structure
    def copy(self) -> "Scene":
        return Scene(self.objects, self.locs, dict(self.on), self.version, self.scene_id,
                     self.min_confidence, self.occluded, self.explicit_goals, self.task, self.blockers)

    @property
    def general(self) -> bool:
        """True if the fast sorting planner is not enough: non-zone goals, constraints,
        or collision blockers. The general task planner handles these."""
        return (self.task is not None and not self.task.simple) or bool(self.blockers)

    def sorts_by_colour(self) -> bool:
        return self.task.sort if self.task is not None else self.explicit_goals is None

    def above(self) -> dict[str, str]:
        """support -> object directly on it (objects only; one object per top)."""
        return {s: o for o, s in self.on.items() if s in self.objects}

    def clear(self) -> set[str]:
        covered = {s for s in self.on.values() if s in self.objects}
        return {o for o in self.objects if o not in covered and self.on[o] != HELD}

    def held(self) -> str | None:
        for o, s in self.on.items():
            if s == HELD:
                return o
        return None

    def base(self, o: str) -> str:
        """Location at the bottom of o's stack (HELD if in the gripper)."""
        s = self.on[o]
        while s in self.objects:
            s = self.on[s]
        return s

    def floor_used(self, loc: str) -> int:
        return sum(1 for s in self.on.values() if s == loc)

    def floor_free(self, loc: str) -> int:
        return self.locs[loc].capacity - self.floor_used(loc)

    def zone_of(self, o: str) -> str | None:
        """The zone with this object's colour, if any (ignores trust / reachability)."""
        c = self.objects[o].color
        if c is None:
            return None
        for l in self.locs.values():
            if l.kind == "zone" and l.color == c:
                return l.id
        return None

    def target(self, o: str) -> str | None:
        """Where this object should end up: the explicit goal if goals were given,
        otherwise the zone of its colour."""
        if self.task is not None:
            return self.task.zone_goals(self).get(o)
        if self.explicit_goals is not None:
            return self.explicit_goals.get(o)
        return self.zone_of(o)

    def trusted(self, o: str) -> bool:
        ob = self.objects[o]
        return ob.color is not None and ob.confidence >= self.min_confidence

    def movable(self, o: str) -> bool:
        """May the server put this object into a plan at all?"""
        ob = self.objects[o]
        return (ob.reachable and ob.pose_confidence >= self.min_confidence
                and o not in self.occluded)

    def frozen(self) -> set[str]:
        """Objects that cannot move in any plan: not movable, or underneath such an object."""
        out: set[str] = set()
        for o in self.objects:
            if not self.movable(o):
                s = o
                while s in self.objects and s not in out:
                    out.add(s)
                    s = self.on[s]
        return out

    def stacks(self) -> list[tuple[str, list[str]]]:
        """[(base location, [objects bottom..top])] for every stack."""
        up = self.above()
        out = []
        for o, s in self.on.items():
            if s not in self.objects:
                col, cur = [o], o
                while cur in up:
                    cur = up[cur]
                    col.append(cur)
                out.append((s, col))
        return out

    # ------------------------------------------------------------ goal
    def goals(self) -> dict[str, str]:
        """object -> zone, only for objects the server is allowed to plan for.

        Unidentified objects have no goal (their colour is not trusted) and frozen
        objects have none either (they cannot be moved); both are reported by
        deferred() so the caller can ask for more information.
        """
        fr = self.frozen()
        if self.task is not None:
            return {o: z for o, z in self.task.zone_goals(self).items() if o not in fr}
        if self.explicit_goals is not None:          # named by id: colour trust is not needed
            return {o: z for o, z in self.explicit_goals.items() if o not in fr}
        return {o: z for o in self.objects
                if o not in fr and self.trusted(o) and (z := self.zone_of(o)) is not None}

    def deferred(self) -> dict[str, str]:
        """Objects left out of the plan, with the reason."""
        out, fr = {}, self.frozen()
        for o, ob in self.objects.items():
            if not self.trusted(o) and self.sorts_by_colour():
                out[o] = "unidentified" if ob.color is None else "low_identity_confidence"
            elif o in self.occluded:
                out[o] = "occluded"
            elif not ob.reachable:
                out[o] = "unreachable"
            elif ob.pose_confidence < self.min_confidence:
                out[o] = "pose_unverified"
            elif o in fr and (z := self.target(o)) is not None and self.base(o) != z:
                out[o] = "blocked_by_unverified_object"
        return out

    def settled(self, goals: dict[str, str] | None = None) -> set[str]:
        """Objects already final: own goal satisfied and nothing below has to move."""
        goals = self.goals() if goals is None else goals
        done: set[str] = set()
        for base, col in self.stacks():
            if base == HELD:
                continue
            for o in col:
                g = goals.get(o)
                if g is not None and g != base:
                    break
                done.add(o)
        return done

    def unsatisfied(self, goals: dict[str, str] | None = None) -> list[str]:
        if self.task is not None and not self.task.simple:
            return self.task.unmet(self)
        goals = self.goals() if goals is None else goals
        return sorted(o for o, z in goals.items() if self.base(o) != z)

    def goal_reached(self) -> bool:
        return self.held() is None and not self.unsatisfied()


# ---------------------------------------------------------------- consistency

def lint(scene: Scene) -> list[str]:
    """Physical consistency checks. An inconsistent scene is rejected, not planned on."""
    issues = []
    tops: dict[str, str] = {}
    for o, s in scene.on.items():
        if o not in scene.objects:
            issues.append(f"support given for unknown object {o}")
        if s == HELD:
            continue
        if s in scene.objects:
            if s == o:
                issues.append(f"{o} is on itself")
            if s in tops:
                issues.append(f"{s} carries both {tops[s]} and {o}")
            tops[s] = o
            if not scene.objects[s].can_support:
                issues.append(f"{o} rests on {s}, which cannot support objects")
        elif s not in scene.locs:
            issues.append(f"{o} rests on unknown support {s}")
    for o in scene.objects:
        if o not in scene.on:
            issues.append(f"{o} has no support")
    if sum(1 for s in scene.on.values() if s == HELD) > 1:
        issues.append("more than one object in the gripper")
    for o in scene.objects:                      # cycles
        seen, s = {o}, scene.on.get(o)
        while s in scene.objects:
            if s in seen:
                issues.append(f"support cycle through {o}")
                break
            seen.add(s)
            s = scene.on.get(s)
    for l in scene.locs.values():
        if l.kind != "table" and scene.floor_used(l.id) > l.capacity:
            issues.append(f"{l.id} holds {scene.floor_used(l.id)} stacks, capacity {l.capacity}")
    colours = [l.color for l in scene.locs.values() if l.kind == "zone"]
    if len(colours) != len(set(colours)):
        issues.append("two zones share a colour")
    return issues


# ---------------------------------------------------------------- parsing

def parse_scene(d: dict) -> Scene:
    """Build a Scene from the wire format (see README, 'Scene message')."""
    objects: dict[str, Obj] = {}
    on: dict[str, str] = {}
    for o in d.get("objects", []):
        t = o.get("type", "cube")
        objects[o["id"]] = Obj(
            id=o["id"], type=t, color=o.get("color"),
            can_support=o.get("can_support", t in FLAT_TOP),
            confidence=float(o.get("confidence", 1.0)),
            pose_confidence=float(o.get("pose_confidence", 1.0)),
            reachable=bool(o.get("reachable", True)),
            position=tuple(o["position"]) if o.get("position") else None)
        if "on" in o:
            on[o["id"]] = HELD if o["on"] in ("gripper", HELD) else o["on"]
    locs: dict[str, Loc] = {TABLE: Loc(TABLE, "table", None, 10 ** 6)}
    if d.get("free_table"):          # the robot may put objects down anywhere free on the table
        locs[TABLE] = Loc(TABLE, "buffer", None, 10 ** 6)
    for z in d.get("zones", []):
        locs[z["id"]] = Loc(z["id"], "zone", z["color"], int(z.get("capacity", 1)))
    for b in d.get("free_slots", []):
        bid = b if isinstance(b, str) else b["id"]
        locs[bid] = Loc(bid, "buffer", None, 1)
    for r in d.get("relations", []):                 # alternative way to give supports
        if r["type"].upper() in ("ON_TOP_OF", "ON"):
            on[r["subject"]] = r["object"]
        elif r["type"].upper() in ("IN", "IN_ZONE", "AT"):
            on.setdefault(r["subject"], r["object"])
    if d.get("holding"):
        on[d["holding"]] = HELD
    for o in objects:
        on.setdefault(o, TABLE)
    blockers = {}
    for o in d.get("objects", []):                   # "blocked_by": collision risk when grasping o
        bs = [b for b in o.get("blocked_by", []) if b in objects and b != o["id"]]
        if bs:
            blockers[o["id"]] = tuple((b, on[b]) for b in bs)
    task = parse_task(d.get("task"), objects, locs)
    goal = d.get("goal") if task is None else None
    explicit = parse_goal(goal, objects, locs)
    if task is not None and task.simple and not task.sort:
        explicit = {a[1]: a[2] for a in task.atoms}
    scene = Scene(objects, locs, on, int(d.get("scene_version", 0)),
                  d.get("scene_id", "scene"), float(d.get("min_confidence", 0.8)),
                  tuple(d.get("occluded", [])), explicit, task, blockers)
    issues = lint(scene)
    if issues:
        raise SceneError("; ".join(issues))
    return scene


def parse_goal(goal, objects: dict, locs: dict) -> dict[str, str] | None:
    """Explicit goals. Accepted forms:
        {"object": "cube_1", "destination": "red_zone"}       one goal
        [{"object": ..., "destination": ...}, ...]             several goals
        {"cube_1": "red_zone", "cyl_1": "blue_zone"}           mapping
    None or missing means the default task: every object into the zone of its colour.
    """
    if goal is None:
        return None
    if isinstance(goal, dict) and "object" in goal and "destination" in goal:
        goal = [goal]
    pairs = [(g["object"], g["destination"]) for g in goal] if isinstance(goal, list) \
        else list(goal.items())
    out = {}
    for o, z in pairs:
        if o not in objects:
            raise SceneError(f"goal names unknown object {o}")
        if z not in locs or locs[z].kind != "zone":
            raise SceneError(f"goal destination {z} is not a zone")
        out[o] = z
    return out


def dump_scene(scene: Scene) -> dict:
    """Inverse of parse_scene (compact wire format)."""
    objs = []
    for o in scene.objects.values():
        e = {"id": o.id, "type": o.type, "color": o.color,
             "on": "gripper" if scene.on[o.id] == HELD else scene.on[o.id]}
        if o.confidence != 1.0:
            e["confidence"] = o.confidence
        if o.pose_confidence != 1.0:
            e["pose_confidence"] = o.pose_confidence
        if not o.reachable:
            e["reachable"] = False
        if o.can_support != (o.type in FLAT_TOP):
            e["can_support"] = o.can_support
        if o.position:
            e["position"] = list(o.position)
        if o.id in scene.blockers:
            e["blocked_by"] = [b for b, where in scene.blockers[o.id] if scene.on.get(b) == where]
        objs.append(e)
    extra = {"goal": dict(scene.explicit_goals)} if scene.explicit_goals is not None else {}
    if scene.task is not None:
        extra = {"task": scene.task.raw}
    return {
        **extra,
        "scene_id": scene.scene_id, "scene_version": scene.version, "objects": objs,
        "zones": [{"id": l.id, "color": l.color, "capacity": l.capacity}
                  for l in scene.locs.values() if l.kind == "zone"],
        "free_slots": [l.id for l in scene.locs.values() if l.kind == "buffer" and l.id != TABLE],
        **({"free_table": True} if scene.locs[TABLE].kind == "buffer" else {}),
    }


# ---------------------------------------------------------------- action model

@dataclass(frozen=True)
class Move:
    """One pick-and-place: take a clear object and put it on a location or object."""
    obj: str
    dst: str

    def __str__(self) -> str:
        return f"MOVE({self.obj} -> {self.dst})"


def why_not_place(scene: Scene, obj: str, dst: str) -> str | None:
    """Reason the destination is illegal for obj, or None if it is legal."""
    if dst == obj:
        return "destination is the object itself"
    if dst in scene.objects:
        d = scene.objects[dst]
        if not d.can_support:
            return f"{dst} cannot support objects"
        if scene.on[dst] == HELD:
            return f"{dst} is in the gripper"
        if dst not in scene.clear():
            return f"{dst} is not clear"
        if not scene.movable(dst):
            return f"pose of {dst} is not verified"
        return None
    if dst not in scene.locs:
        return f"unknown destination {dst}"
    if scene.locs[dst].kind == "table":
        return "the open table is not a verified place target"
    if scene.floor_free(dst) <= 0:
        return f"{dst} has no free ground pose"
    return None


def why_not_place_task(scene: Scene, obj: str, dst: str) -> str | None:
    """Physical legality first, then the task's constraints."""
    reason = why_not_place(scene, obj, dst)
    if reason is None and scene.task is not None and scene.task.constrained:
        reason = scene.task.violation(scene, obj, dst)
    return reason


def why_not(scene: Scene, m: Move) -> str | None:
    """Precondition check for a Move. Returns the violated precondition or None."""
    if m.obj not in scene.objects:
        return f"unknown object {m.obj}"
    held = scene.held()
    if held is not None and held != m.obj:
        return f"gripper already holds {held}"
    if held is None and m.obj not in scene.clear():
        return f"{m.obj} is not clear"
    if not scene.movable(m.obj):
        return f"{m.obj} is unreachable or its pose is unverified"
    if scene.task is not None and m.obj in scene.task.keep:
        return f"the task says {m.obj} must not be moved"
    if held is None:
        for blocker, where in scene.blockers.get(m.obj, ()):
            if scene.on.get(blocker) == where:
                return f"collision risk: {blocker} must be moved before {m.obj} can be picked"
    return why_not_place_task(scene, m.obj, m.dst)


def apply(scene: Scene, m: Move) -> None:
    """Apply the effect of a Move in place (preconditions are the caller's job)."""
    scene.on[m.obj] = m.dst


def legal_dests(scene: Scene, obj: str) -> list[str]:
    out = [l.id for l in scene.locs.values()
           if l.kind != "table" and scene.floor_free(l.id) > 0]
    out += [o for o in sorted(scene.clear())
            if o != obj and scene.objects[o].can_support and scene.movable(o)]
    if scene.task is not None and scene.task.constrained:
        out = [d for d in out if scene.task.violation(scene, obj, d) is None]
    return out


# ---------------------------------------------------------------- validation

@dataclass
class Validation:
    ok: bool
    reason: str = ""
    failed_step: int | None = None       # index of the first illegal step
    valid_prefix: int = 0                # number of leading steps that are legal
    goal_reached: bool = False
    final: Scene | None = None


def validate(scene: Scene, plan: Iterable[Move], require_goal: bool = True) -> Validation:
    """Replay a plan on a copy of the scene, checking every precondition.

    This is the gate in front of the robot: plans from any source (planner, case
    base, a language model) are executed only if they pass here against the
    *current* scene version.
    """
    s = scene.copy()
    n = 0
    for i, m in enumerate(plan):
        reason = why_not(s, m)
        if reason:
            return Validation(False, f"step {i} {m}: {reason}", i, n, False, s)
        apply(s, m)
        n += 1
    done = s.goal_reached()
    if require_goal and not done:
        return Validation(False, "plan is executable but does not reach the goal: "
                          + ", ".join(s.unsatisfied()), None, n, False, s)
    return Validation(True, "", None, n, done, s)


# ---------------------------------------------------------------- controller contract

def to_contract(scene: Scene, plan: list[Move]) -> list[dict]:
    """Expand moves into the PICK / PLACE vocabulary agreed with the robot controller.

    Every step carries the facts the laptop should observe afterwards ("expect"),
    so the executor can verify each outcome instead of assuming it.
    """
    s = scene.copy()
    steps: list[dict] = []
    for m in plan:
        src = s.on[m.obj]
        if src != HELD:
            steps.append({"step": len(steps), "action": "PICK", "object": m.obj,
                          "from": src, "expect": {"holding": m.obj}})
        kind = "object" if m.dst in s.objects else ("table" if m.dst == TABLE else s.locs[m.dst].kind)
        steps.append({
            "step": len(steps),
            "action": "MOVE_TO_TEMPORARY_LOCATION" if kind == "buffer" else "PLACE",
            "object": m.obj, "destination": m.dst, "destination_kind": kind,
            "expect": {"on": {m.obj: m.dst}, "holding": None}})
        apply(s, m)
    return steps
