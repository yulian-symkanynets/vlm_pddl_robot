"""Task understanding: from an instruction in words to a validated task specification.

This is the one job on the server that needs a language model. Everything after
it (planning, validation, replanning) is symbolic, so the model is called at
most once per task and never during replanning.

    instruction + object list
        -> cache hit?                  same instruction, same kinds of objects: reuse
        -> rule parser                 common phrasings, no model, microseconds
        -> language model              prompt = knowledge base + retrieved examples (RAG) + objects
             -> parse_task() validates the answer against the scene
             -> on error: the error goes back to the model (up to max_rounds)
        -> "clarify" with a question   if nothing produced a valid task

The model also sees the CURRENT arrangement, clearly labelled, because relative
instructions ("swap the two stacks", "reverse it", "take red off yellow") cannot be
understood without it. The earlier failure (the model copied the present state as the
goal) is caught by the validation instead: an answer that is already true in the current
scene is sent back once as "nothing would change".
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .llm import LLM, extract_json
from semplan.scene import Scene
from semplan.task import TaskError, parse_task

KB_PATH = Path(__file__).parent / "knowledge_base.json"
MEMORY_PATH = Path(__file__).parent / "results" / "task_examples.json"


def load_kb() -> dict:
    return json.loads(KB_PATH.read_text())


@dataclass
class Understanding:
    task: dict | None                 # validated task message, or None
    source: str                       # structured | cache | rules | llm | none
    question: str = ""                # set when the instruction needs clarification
    llm_calls: int = 0
    rounds: int = 0
    ms: float = 0.0
    examples_used: int = 0
    errors: list[str] = field(default_factory=list)


# ================================================================ naming objects in words

COLOURS = {"red", "blue", "green", "yellow", "orange", "purple", "black", "white", "pink", "brown", "grey", "gray"}
SHAPES = {"cube": "cube", "cubes": "cube", "block": "cube", "blocks": "cube", "box": "cube",
          "cylinder": "cylinder", "cylinders": "cylinder", "prism": "prism", "prisms": "prism",
          "object": None, "objects": None, "thing": None, "things": None, "one": None, "ones": None}
EVERYTHING = {"everything", "all", "every", "rest", "others", "remaining"}


def describe_objects(scene: Scene) -> list[str]:
    return [f"- {o}: {ob.color or 'unknown colour'} {ob.type}" for o, ob in sorted(scene.objects.items())]


def describe_arrangement(scene: Scene) -> list[str]:
    """Current stacks, bottom to top, for relative instructions."""
    out = []
    for base, col in sorted(scene.stacks(), key=lambda bc: (bc[0], bc[1])):
        where = "in the gripper" if base == "__gripper__" else ("on the table" if base == "table" else f"in {base}")
        out.append(f"- {where}: " + (" -> ".join(col) if len(col) > 1 else col[0])
                   + ("   (bottom -> top)" if len(col) > 1 else ""))
    return out


def resolve(phrase: str, scene: Scene) -> tuple[list[str], str]:
    """Objects a phrase refers to. -> (ids, problem). Exact ids win; otherwise colour + shape."""
    words = re.findall(r"[a-z0-9_]+", phrase.lower())
    ids = [w for w in words if w in scene.objects]
    if ids:
        return ids, ""
    joined = "_".join(words)
    close = [o for o in scene.objects if o.lower() == joined or o.lower().replace("_", " ") in phrase.lower()]
    if close:
        return close, ""
    colour = next((w for w in words if w in COLOURS), None)
    shape_words = [w for w in words if w in SHAPES]
    shape = SHAPES[shape_words[0]] if shape_words else None
    plural = any(w.endswith("s") and w in SHAPES for w in words) or bool(set(words) & EVERYTHING)
    if colour is None and shape is None and not (set(words) & EVERYTHING):
        return [], f"cannot tell which object '{phrase.strip()}' means"
    match = [o for o, ob in sorted(scene.objects.items())
             if (colour is None or ob.color == colour) and (shape is None or ob.type == shape)]
    if not match:
        return [], f"there is no {re.sub('^(the|a|an) ', '', phrase.strip())} in the scene"
    if len(match) > 1 and not plural:
        return [], f"'{phrase.strip()}' could be {', '.join(match)}; which one?"
    return match, ""


def resolve_place(phrase: str, scene: Scene) -> tuple[str | None, str]:
    """A zone, slot, 'table' or object named by a phrase."""
    low = phrase.lower()
    words = re.findall(r"[a-z0-9_]+", low)
    for l in scene.locs.values():
        if l.id in words or l.id.replace("_", " ") in low:
            return l.id, ""
    if re.search(r"\b(zone|area|paper|place|spot|field|mat)\b", low):
        colour = next((w for w in words if w in COLOURS), None)
        z = [l.id for l in scene.locs.values() if l.kind == "zone" and l.color == colour]
        if z:
            return z[0], ""
        return None, f"there is no {colour or ''} zone"
    if re.search(r"\b(table|desk)\b", low):
        return "table", ""
    ids, problem = resolve(phrase, scene)
    if len(ids) == 1:
        return ids[0], ""
    return None, problem or f"'{phrase.strip()}' is not one place"


# ================================================================ rule parser

_ART = r"(?:the|a|an|all|all the|every|each)\s+"
_NEG = r"(?:do not|don't|dont|never|without|no|not|avoid)"


class RuleInterpreter:
    """Understands common English phrasings without a model. Returns None when any part of
    the instruction is left unexplained, so that the language model can take over."""

    name = "rules"

    def interpret(self, instruction: str, scene: Scene) -> tuple[dict | None, str]:
        text = " " + re.sub(r"\s+", " ", instruction.lower().strip().rstrip(".!?")) + " "
        text = text.replace("on top of", "on").replace("onto", "on").replace("into", "in").replace(" inside ", " in ")
        # "pick (up) X and place/put it on Y" -> "put X on Y"
        text = re.sub(r"\b(?:pick(?:\s+up)?|grab|take|lift|grasp)\s+(.+?)\s*,?\s*(?:and\s+)?(?:then\s+)?"
                      r"(?:place|put|drop|set|move|stack)\s+(?:it\s+|them\s+)?(on|in|to)\b", r"put \1 \2", text)
        goal, cons, problems = [], {}, []
        verbs = r"put|place|move|stack|sort|leave|keep|use|make|arrange|build|clear"
        strong = r"\s*(?:;|\.|\bbut\b|\band then\b|\bwhile\b|\bso that\b|,?\s*\band\b(?=\s+(?:" + _NEG + "|" + verbs \
            + r"))|,?\s*\bthen\b(?=\s+(?:" + verbs + r"))|,(?=\s*(?:" + _NEG + "|" + verbs + r"))|(?=\bwithout\b))\s*"
        clauses = []
        for chunk in re.split(strong, text):
            chunk = chunk.strip()
            if not chunk:
                continue
            if re.search(r"\b(stack|pile|tower|arrange|order)\b", chunk):
                clauses.append(chunk)                 # a list of objects: keep its commas together
            else:
                clauses += [c.strip() for c in re.split(r"\s*,\s*", chunk) if c.strip()]
        for c in clauses:
            if re.fullmatch(r"(?:please\s+)?(?:rearrange|reorder|arrange|move|change)\s+(?:them|it|the objects|everything|things)", c):
                continue                              # filler: the goal follows in another clause
            handled = self._constraint(c, scene, cons, problems) or self._goal(c, scene, goal, problems)
            if not handled:
                return None, f"did not understand: '{c}'"
        if problems:
            return None, problems[0]
        if not goal:
            return None, "the instruction gives no goal"
        task = {"goal": goal}
        if cons:
            task["constraints"] = cons
        return task, ""

    def _constraint(self, c, scene, cons, problems) -> bool:
        if not re.search(r"^(?:and |please )?" + _NEG + r"\b|\b" + _NEG + r"\b.*\b(table|temporary|slot|touch|move|put|place|stack)", c) \
                and not re.search(r"\b(leave|keep)\b.*\b(where|alone|in place|untouched)\b|\bat most\b|\bno higher\b|\bmaximum\b", c):
            return False
        m = re.search(r"\b(?:at most|maximum|max|no higher than|not higher than)\s+(\d+|one|two|three)", c)
        if m:
            cons["max_stack_height"] = {"one": 1, "two": 2, "three": 3}.get(m.group(1)) or int(m.group(1))
            return True
        m = re.search(_NEG + r"\s+(?:put|place|stack|set)\s+(.+?)\s+(?:on|in)\s+(.+)$", c)
        if m and not re.search(r"\b(table|temporary|slot|desk)\b", m.group(2)):
            subj = m.group(1).strip()
            objs = ["*"] if subj in ("anything", "it", "them", "objects", "any object", "things") else resolve(subj, scene)[0]
            place, problem = resolve_place(m.group(2), scene)
            if not objs or place is None:
                problems.append(problem or f"cannot tell what '{subj}' means")
                return True
            cons.setdefault("never_on", []).extend([[o, place] for o in objs])
            return True
        if re.search(r"\b(table|temporary|slot|slots|desk|parking|aside)\b", c):
            cons["no_temporary"] = True
            return True
        m = re.search(r"(?:" + _NEG + r"\s+(?:move|touch|use)|leave|keep)\s+(.+?)(?:\s+(?:where|alone|in place|untouched).*)?$", c)
        if m:
            ids, problem = resolve(m.group(1), scene)
            if not ids:
                problems.append(problem)
                return True
            cons.setdefault("keep", []).extend(ids)
            return True
        return False

    def _goal(self, c, scene, goal, problems) -> bool:
        if re.search(r"\bsort\b|\bby colou?r\b|\btidy\b|\bclean up\b|\bwhere (?:it|they) belongs?\b", c):
            goal.append({"sort_by_color": True})
            return True
        m = re.search(r"\b(?:stack|pile|build a tower(?: of| with)?|arrange|order)\b\s*(.+)$", c)
        if m and re.search(r"\b(bottom|top|order|tower|stack|pile)\b", c):
            body = m.group(1)
            zone = None
            mz = re.search(r"\s+in\s+((?:the\s+)?\w+\s+(?:zone|area|paper|place))", body)
            if mz:
                zone, problem = resolve_place(mz.group(1), scene)
                body = body.replace(mz.group(0), " ")
                if zone is None:
                    problems.append(problem)
            reverse = bool(re.search(r"top to bottom|from the top|top first", body))
            body = re.sub(r"\b(from|in this order|in that order|in order|bottom to top|top to bottom|from the bottom|"
                          r"from the top|bottom first|top first|as a tower|a tower|tower|:)\b", " ", body)
            parts = [p.strip() for p in re.split(r"\s*(?:,|\bthen\b|\band\b|\bon\b(?!\s+top)|>|\bunder\b)\s*", body) if p.strip()]
            under_form = " on " in (" " + m.group(1) + " ") and "," not in m.group(1)
            chain = []
            for p in parts:
                ids, problem = resolve(p, scene)
                if len(ids) != 1:
                    problems.append(problem or f"'{p}' is not one object")
                    return True
                chain.append(ids[0])
            if under_form or reverse:                # "stack a on b on c": first is the top
                chain = chain[::-1]
            if len(chain) < 2:
                return False
            g = {"stack": chain}
            if zone:
                g["in"] = zone
            goal.append(g)
            return True
        m = re.search(r"\b(?:move|take|get|lift|remove|pick)\s+(.+?)\s+(?:off|off of|from)\s+(.+)$", c)
        if m and not re.search(r"\b(?:and|then)\s+(?:put|place|move|set)\b", m.group(2)):
            ids, problem = resolve(m.group(1), scene)
            below, problem2 = resolve(m.group(2), scene)
            if len(ids) != 1 or len(below) != 1:
                problems.append(problem or problem2 or "'off' needs one object and one object")
                return True
            goal.append({"object": ids[0], "not_on": below[0]})
            return True
        # Relations without "put", or several in one clause:
        #   "make blue on yellow and red under green", "blue on yellow, red under green",
        #   "I want the blue cube on the yellow one"
        rel = r"\s(?:on|in|to|under|below|beneath)\s"
        body = re.sub(r"^(?:please\s+)?(?:(?:i|we)\s+(?:want|need)|make|have|let|get|arrange it so that|so that)\s+", "", c)
        parts = [p.strip() for p in re.split(r"\s*(?:,|\band\b)\s*", body) if p.strip()]
        put_verb = re.match(r"(?:please\s+)?(?:put|place|move|set|bring|take|stack)\b", c)
        if parts and all(re.search(rel, f" {p} ") for p in parts) and (len(parts) > 1 or not put_verb):
            for p in parts:
                p = re.sub(r"^(?:put|place|move|set|bring|take|stack)\s+", "", p)
                p = re.sub(r"\s+(?:is|are|sits|stands|lies|should be|must be)\s+(?=(?:on|in|to|under|below|beneath)\s)", " ", p)
                if not self._goal("put " + p, scene, goal, problems):
                    return False
            return True
        m = re.search(r"\b(?:put|place|move|set|bring|take|stack|get)\s+(.+?)\s+(on|in|to|under|below|beneath)\s+(.+)$", c)
        if m:
            subj, prep, dest = m.group(1), m.group(2), m.group(3)
            ids, problem = resolve(subj, scene)
            place, problem2 = resolve_place(dest, scene)
            if not ids or place is None:
                problems.append(problem or problem2)
                return True
            if prep in ("under", "below", "beneath"):
                if len(ids) != 1 or place not in scene.objects:
                    problems.append("'under' needs one object and one object")
                    return True
                goal.append({"object": place, "on": ids[0]})
                return True
            zone = place in scene.locs and scene.locs[place].kind == "zone"
            if not zone and place != "table" and len(ids) > 1:
                problems.append(f"only one object can rest directly on {place}")
                return True
            for o in ids:
                goal.append({"object": o, "in" if zone else "on": place})
            return True
        m = re.search(r"\b(?:nothing|no object|none)\s+(?:is|are|may be|stays?|remains?|sits)?\s*(?:left\s+)?on\s+(.+)$", c)
        if m:
            ids, problem = resolve(m.group(1), scene)
            if not ids:
                problems.append(problem)
                return True
            goal.extend({"object": o, "clear": True} for o in ids)
            return True
        m = re.search(r"^(.+?)\s+(?:is|are|ends up|sits|stands|lies)\s+(on|in|under|below|beneath)\s+(.+)$", c)
        if m:
            a, _ = resolve(m.group(1), scene)
            place, problem = resolve_place(m.group(3), scene)
            if len(a) != 1 or place is None:
                problems.append(problem or f"'{m.group(1)}' is not one object")
                return True
            if m.group(2) in ("under", "below", "beneath"):
                goal.append({"object": place, "on": a[0]})
            else:
                zone = place in scene.locs and scene.locs[place].kind == "zone"
                goal.append({"object": a[0], "in" if zone else "on": place})
            return True
        m = re.search(r"\b(?:clear|uncover|free)\s+(.+)$", c)
        if m:
            ids, problem = resolve(m.group(1), scene)
            if not ids:
                problems.append(problem)
                return True
            goal.extend({"object": o, "clear": True} for o in ids)
            return True
        return False


# ================================================================ example memory (RAG)

SEED_EXAMPLES = [
    {"instruction": "sort the objects by colour",
     "objects": ["a: red cube", "b: blue cylinder"], "zones": ["red_zone", "blue_zone"],
     "task": {"goal": [{"sort_by_color": True}]}},
    {"instruction": "put the red cube on the blue cylinder",
     "objects": ["a: red cube", "b: blue cylinder"], "zones": ["red_zone", "blue_zone"],
     "task": {"goal": [{"object": "a", "on": "b"}]}},
    {"instruction": "build a tower: green cube at the bottom, then the red cube, the blue cylinder on top, in the blue zone",
     "objects": ["a: red cube", "b: blue cylinder", "c: green cube"], "zones": ["red_zone", "blue_zone"],
     "task": {"goal": [{"stack": ["c", "a", "b"], "in": "blue_zone"}]}},
    {"instruction": "swap the two cubes in the stack but do not put anything on the table",
     "objects": ["a: red cube", "c: green cube"], "zones": ["red_zone", "green_zone"],
     "task": {"goal": [{"object": "a", "on": "c"}], "constraints": {"no_temporary": True}}},
    {"instruction": "tidy up, but leave the green cube where it is and never stack more than two objects",
     "objects": ["a: red cube", "b: blue cylinder", "c: green cube"], "zones": ["red_zone", "blue_zone", "green_zone"],
     "task": {"goal": [{"sort_by_color": True}], "constraints": {"keep": ["c"], "max_stack_height": 2}}},
    {"instruction": "move the blue cylinder to the red zone and make sure nothing is on the red cube",
     "objects": ["a: red cube", "b: blue cylinder"], "zones": ["red_zone", "blue_zone"],
     "task": {"goal": [{"object": "b", "in": "red_zone"}, {"object": "a", "clear": True}]}},
    {"instruction": "roztried predmety podla farby, ale nepouzivaj docasne miesta na stole",
     "objects": ["a: red cube", "b: blue cylinder"], "zones": ["red_zone", "blue_zone"],
     "task": {"goal": [{"sort_by_color": True}], "constraints": {"no_temporary": True}}},
    {"instruction": "reverse the stack with the blue cube",
     "objects": ["a: red cube", "b: blue cube", "c: green cube"], "zones": [],
     "task": {"goal": [{"reverse": "b"}]}},
    {"instruction": "flip both towers upside down",
     "objects": ["a: red cube", "b: blue cube", "c: green cube", "d: yellow cube"], "zones": [],
     "task": {"goal": [{"reverse": "all"}]}},
    {"instruction": "swap the green cube and the red cube",
     "objects": ["a: red cube", "b: blue cube", "c: green cube"], "zones": [],
     "task": {"goal": [{"swap": ["c", "a"]}]}},
    {"instruction": "make two towers: red on blue, and yellow on green",
     "objects": ["a: red cube", "b: blue cube", "c: green cube", "d: yellow cube"], "zones": [],
     "task": {"goal": [{"object": "a", "on": "b"}, {"object": "d", "on": "c"}]}},
    {"instruction": "polož modrú kocku na červenú a zelenú na stôl",
     "objects": ["a: red cube", "b: blue cube", "c: green cube"], "zones": [],
     "task": {"goal": [{"object": "b", "on": "a"}, {"object": "c", "on": "table"}]}},
    {"instruction": "unstack everything",
     "objects": ["a: red cube", "b: blue cube", "c: green cube"], "zones": [],
     "task": {"goal": [{"object": "a", "on": "table"}, {"object": "b", "on": "table"}, {"object": "c", "on": "table"}]}},
    {"instruction": "take the red cube off the blue one",
     "objects": ["a: red cube", "b: blue cube"], "zones": [],
     "task": {"goal": [{"object": "a", "not_on": "b"}]}},
    {"instruction": "blue on yellow and red under green",
     "objects": ["a: red cube", "b: blue cube", "c: green cube", "d: yellow cube"], "zones": [],
     "task": {"goal": [{"object": "b", "on": "d"}, {"object": "c", "on": "a"}]}},
    {"instruction": "vymeň modrú a žltú kocku",
     "objects": ["a: red cube", "b: blue cube", "d: yellow cube"], "zones": [],
     "task": {"goal": [{"swap": ["b", "d"]}]}},
    {"instruction": "rearrange these cubes",
     "objects": ["a: red cube", "b: blue cube"], "zones": [],
     "task": {"goal": [], "question": "How should the cubes be arranged at the end?"}},
]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class ExampleMemory:
    """Solved instructions, retrieved by word overlap and shown to the model as examples.

    Starts with a few hand-written examples and grows with every instruction whose
    plan was executed successfully (TaskService calls add()). Stored as plain JSON.
    Word overlap needs no embedding model; swap `retrieve` for a vector search if
    the collection becomes large or multilingual.
    """

    def __init__(self, path: Path | None = None, seed: bool = True):
        self.path = path
        self.items: list[dict] = list(SEED_EXAMPLES) if seed else []
        if path and path.exists():
            self.items += json.loads(path.read_text())

    def retrieve(self, instruction: str, k: int = 3) -> list[dict]:
        q = _tokens(instruction)
        scored = []
        for it in self.items:
            t = _tokens(it["instruction"])
            scored.append((len(q & t) / ((len(q) * len(t)) ** 0.5 or 1), it))
        scored.sort(key=lambda x: -x[0])
        return [it for score, it in scored[:k] if score > 0]

    def add(self, instruction: str, scene: Scene, task: dict) -> None:
        if any(it["instruction"].lower() == instruction.lower() for it in self.items):
            return
        item = {"instruction": instruction, "objects": [l[2:] for l in describe_objects(scene)],
                "zones": sorted(l.id for l in scene.locs.values() if l.kind == "zone"), "task": task}
        self.items.append(item)
        if self.path:
            self.path.parent.mkdir(exist_ok=True)
            learned = self.items[len(SEED_EXAMPLES):]
            self.path.write_text(json.dumps(learned, indent=1))


# ================================================================ language model interpreter

PROMPT = """You translate an instruction for a robot arm into a task specification.
The robot rearranges objects on a table. You do not plan the moves; a planner does that.

What can be asked for (goal entries):
{goals}

Rules that the plan must respect on the way (constraints, optional):
{constraints}

Facts about the robot:
{rules}
{examples}
Now the real scene.
Objects:
{objects}
CURRENT arrangement (what rests on what right now; describe the FINAL arrangement, not this one):
{arrangement}
Zones: {zones}
Free temporary slots: {slots}

Instruction: {instruction}
{feedback}
Use only the ids listed above. A stack is listed bottom to top. List only what the instruction asks for.
Add constraints ONLY if the instruction explicitly states them; most instructions have none.
If the instruction does not say what the final arrangement should be (for example "rearrange these
cubes", "do something", "put it there"), refers to something that is not listed, is ambiguous, or asks
for something the forms above cannot express, return {{"goal": [], "question": "<one short question for the user>"}}.
Return only JSON: {{"goal": [ ... ], "constraints": {{ ... }}}}"""


def expand_operations(goal, scene: Scene) -> list:
    """Goal entries that are operations on the CURRENT arrangement, chosen by the model and
    computed here exactly:
        {"reverse": "<id of any object in the stack>" | "all"}   the stack upside down
        {"swap": ["<id>", "<id>"]}                                 the two objects change places
    Other entries pass through unchanged."""
    if isinstance(goal, dict):
        goal = [goal]
    out = []
    for g in goal if isinstance(goal, list) else []:
        if isinstance(g, dict) and "reverse" in g:
            r = g["reverse"]
            stacks = [(b, col) for b, col in scene.stacks() if len(col) > 1 and b != "__gripper__"]
            if r in ("all", "*", "everything", True):
                chosen = stacks
            else:
                ids = r if isinstance(r, list) else [r]
                for i in ids:
                    if i not in scene.objects:
                        raise TaskError(f"unknown object {i!r} in reverse")
                chosen = [(b, col) for b, col in stacks if set(ids) & set(col)]
            if not chosen:
                raise TaskError("reverse: that object is not in a stack of two or more objects")
            for b, col in chosen:
                e = {"stack": col[::-1]}
                if b in scene.locs and scene.locs[b].kind == "zone":
                    e["in"] = b
                out.append(e)
        elif isinstance(g, dict) and "swap" in g:
            pair = g["swap"]
            if not (isinstance(pair, list) and len(pair) == 2 and pair[0] != pair[1]
                    and all(x in scene.objects for x in pair)):
                raise TaskError("swap needs a list of two different object ids")
            a, b = pair
            sigma = lambda x: b if x == a else a if x == b else x
            touched = {a, b} | {o for o, sup in scene.on.items() if sup in (a, b)}
            for o in sorted(touched):
                t = sigma(scene.on[sigma(o)])
                if t == "__gripper__":
                    continue
                if t in scene.objects or (t in scene.locs and scene.locs[t].kind == "buffer" and t != "table"):
                    out.append({"object": o, "on": t})
                elif t in scene.locs and scene.locs[t].kind == "zone":
                    out.append({"object": o, "in": t})
                else:
                    out.append({"object": o, "on": "table"})
        else:
            out.append(g)
    return out


# Words that must appear in an instruction before a constraint of that kind is believed.
CONSTRAINT_WORDS = {
    "no_temporary": ("table", "temporar", "aside", "park", "slot", "desk", "stôl", "stol", "stůl", "odkla"),
    "keep": ("keep", "leave", "stay", "where it is", "don't move", "do not move", "not move", "touch",
             "nehýb", "nehyb", "nechaj", "nech "),
    "never_on": ("never", "not on", "don't put", "do not put", "avoid", "nikdy", "nedávaj", "nedavaj"),
    "max_stack_height": ("high", "tall", "height", "at most", "max", "level", "vysok", "najviac"),
}


def drop_unrequested(task: dict, parsed, instruction: str, scene: Scene) -> tuple[dict, list[str]]:
    """Remove constraints whose kind the instruction never mentions, when they make the task
    impossible (small models like to add "no_temporary" on their own)."""
    cons = dict(task.get("constraints") or {})
    low = instruction.lower()
    unasked = [k for k in cons if k in CONSTRAINT_WORDS and not any(w in low for w in CONSTRAINT_WORDS[k])
               and cons[k] not in (False, None, [], 0)]
    if not unasked or not infeasible(task, scene):
        return task, []
    for k in unasked:
        cons.pop(k)
    out = {"goal": task["goal"]}
    if cons:
        out["constraints"] = cons
    return out, unasked


def infeasible(task: dict, scene: Scene, ms: float = 150.0) -> str | None:
    """Reason why no plan exists for this task on this scene, or None if one was found."""
    from semplan.task_planner import plan_task
    s = scene.copy()
    s.task = parse_task(json.loads(json.dumps(task)), scene.objects, scene.locs)
    s.explicit_goals = None
    r = plan_task(s, ms)
    return r.reason or "no plan" if r.status == "infeasible" else None


class LLMInterpreter:
    """Instruction -> task with a language model, retrieved examples and a validation loop."""

    name = "llm"

    def __init__(self, llm: LLM, memory: ExampleMemory | None = None, max_rounds: int = 3, k: int = 3):
        self.llm, self.memory, self.max_rounds, self.k = llm, memory, max_rounds, k
        self.kb = load_kb()

    def prompt(self, instruction: str, scene: Scene, feedback: str = "") -> tuple[str, int]:
        kb = self.kb
        examples = self.memory.retrieve(instruction, self.k) if self.memory else []
        ex = ""
        if examples:
            ex = "\nExamples of earlier instructions and their correct translation:\n" + "\n".join(
                f"Objects: {'; '.join(e['objects'])}. Zones: {', '.join(e['zones'])}.\n"
                f"Instruction: {e['instruction']}\n{json.dumps(e['task'])}\n" for e in examples)
        text = PROMPT.format(
            goals="\n".join(f"  {json.dumps(g['form'])}   {g['meaning']}" for g in kb["goal_vocabulary"]),
            constraints="\n".join(f"  {json.dumps(c['form'])}   {c['meaning']}" for c in kb["constraint_vocabulary"]),
            rules="\n".join(f"- {r}" for r in kb["rules"]),
            examples=ex, objects="\n".join(describe_objects(scene)),
            arrangement="\n".join(describe_arrangement(scene)),
            zones=", ".join(sorted(l.id for l in scene.locs.values() if l.kind == "zone")) or "(none)",
            slots=("free table space (destination 'table')" if scene.locs["table"].kind == "buffer" else
                   ", ".join(sorted(l.id for l in scene.locs.values() if l.kind == "buffer")) or "(none)"),
            instruction=instruction, feedback=feedback)
        return text, len(examples)

    def interpret(self, instruction: str, scene: Scene) -> Understanding:
        before, feedback, errors, used = self.llm.calls, "", [], 0
        for round_no in range(1, self.max_rounds + 1):
            prompt, used = self.prompt(instruction, scene, feedback)
            answer = self.llm.complete(prompt)
            js = extract_json(answer)
            if not isinstance(js, dict) or "goal" not in js:
                err = "the answer was not JSON with a 'goal' list"
            elif js.get("question") and not js.get("goal"):
                return Understanding(None, "llm", str(js["question"]), self.llm.calls - before,
                                     round_no, examples_used=used, errors=errors)
            else:
                task = {"goal": js["goal"]}
                if js.get("constraints"):
                    task["constraints"] = js["constraints"]
                try:
                    task["goal"] = expand_operations(task["goal"], scene)
                    parsed = parse_task(json.loads(json.dumps(task)), scene.objects, scene.locs)
                    if parsed is not None and not parsed.sort and parsed.atoms and not parsed.unmet(scene):
                        if round_no == self.max_rounds:     # the model keeps describing the present
                            return Understanding(None, "llm", "What should the final arrangement look like?",
                                                 self.llm.calls - before, round_no, examples_used=used,
                                                 errors=errors + ["goal already true"])
                        raise TaskError("this goal is already true in the CURRENT arrangement, so nothing "
                                        "would change; describe the FINAL arrangement the instruction asks for, "
                                        "or ask a question if the instruction does not say it")
                    task, dropped = drop_unrequested(task, parsed, instruction, scene)
                    if dropped:
                        errors.append("dropped constraints the instruction does not state: " + ", ".join(dropped))
                    problem = infeasible(task, scene)
                    if problem and round_no < self.max_rounds:
                        raise TaskError(f"no sequence of moves can achieve this task ({problem}); "
                                        "check that it states exactly what the instruction asks for")
                    return Understanding(task, "llm", "", self.llm.calls - before, round_no,
                                         examples_used=used, errors=errors)
                except (TaskError, KeyError, TypeError, ValueError) as e:
                    err = str(e) or type(e).__name__
            errors.append(err)
            feedback = (f"\nYour previous answer was rejected: {err}\n"
                        f"Previous answer: {str(answer)[:400]}\nCorrect it.\n")
        return Understanding(None, "llm", f"could not turn the instruction into a valid task ({errors[-1]})",
                             self.llm.calls - before, self.max_rounds, examples_used=used, errors=errors)


# ================================================================ the chain

class TaskInterpreter:
    """cache -> rules -> model. `order` chooses which stages run and in which order:
    "rules,llm" (default: model only when the rules give up), "llm" (always the model),
    "rules" (never a model), "llm,rules"."""

    def __init__(self, llm: LLM | None = None, memory: ExampleMemory | None = None,
                 order: str = "rules,llm", use_cache: bool = True, max_rounds: int = 3):
        self.memory = memory if memory is not None else ExampleMemory()
        self.rules = RuleInterpreter()
        self.llm = LLMInterpreter(llm, self.memory, max_rounds) if llm is not None else None
        self.order = [s for s in order.split(",") if s in ("rules", "llm")]
        self.use_cache = use_cache
        self.cache: dict[tuple, dict] = {}
        self.stats = {"cache": 0, "rules": 0, "llm": 0, "none": 0, "llm_calls": 0}

    @staticmethod
    def _cache_key(instruction: str, scene: Scene) -> tuple:
        objs = tuple(sorted((o, ob.color, ob.type) for o, ob in scene.objects.items()))
        return (re.sub(r"\s+", " ", instruction.lower().strip()), objs, tuple(sorted(scene.locs)))

    def understand(self, instruction: str, scene: Scene) -> Understanding:
        t0 = time.perf_counter()
        key = self._cache_key(instruction, scene)
        u = None
        if self.use_cache and key in self.cache:
            u = Understanding(self.cache[key], "cache")
        question = ""
        for stage in ([] if u else self.order):
            if stage == "rules":
                task, problem = self.rules.interpret(instruction, scene)
                if task is not None:
                    try:
                        parse_task(json.loads(json.dumps(task)), scene.objects, scene.locs)
                        u = Understanding(task, "rules")
                        break
                    except TaskError as e:
                        problem = str(e)
                question = question or problem
            elif stage == "llm" and self.llm is not None:
                u = self.llm.interpret(instruction, scene)
                if u.task is not None:
                    break
                question, u = u.question or question, (u if self.order[-1] == "llm" else None)
        if u is None:
            u = Understanding(None, "none", question or "the instruction was not understood")
        if u.task is None and not u.question:
            u.question = question or "the instruction was not understood"
        if u.task is not None and self.use_cache:
            self.cache[key] = u.task
        u.ms = (time.perf_counter() - t0) * 1e3
        self.stats[u.source if u.task is not None else "none"] += 1
        self.stats["llm_calls"] += u.llm_calls
        return u
