# Task planning server

The server side of the robot project: it receives a semantic scene and a task,
understands the task, and returns a validated sequence of robot operations. It
keeps the task for the whole session, so when a step fails or the scene changes
it replans without asking the model again.

Everything below was tested on a laptop without a GPU, with a rule parser and
scripted model answers. **No real VLM has been run through this server yet.**

## Start

```bash
pip install -r approaches/requirements.txt

python -m approaches.task_demo                                  # nine instructions on one scene
python -m approaches.task_server                                # no model: rules + structured tasks
python -m approaches.task_server --model qwen3-vl-8b            # GPU machine, key from models.yaml
python -m approaches.task_server --model qwen3-vl-8b --mode vlm_plan    # the model also writes the plan
python -m pytest tests/test_task_server.py -q                   # 21 tests
python -m approaches.model_bench --models qwen3-vl-4b qwen3-vl-8b --modes hybrid vlm_plan
```

Default port is 8020 (8010 is the older `server.py`, 8000 is taken on the lab PC).

## What a task can be

A task is goals plus optional constraints. It arrives as words (`instruction`)
or already structured (`task`). Without either, the task is "sort by colour".

| Goal | Meaning |
|---|---|
| `{"sort_by_color": true}` | every object into the zone of its colour |
| `{"object": "a", "in": "red_zone"}` | a ends in that zone (on the floor or stacked inside it) |
| `{"object": "a", "on": "b"}` | a rests directly on b |
| `{"object": "a", "on": "table"}` | a stands alone on a free slot |
| `{"object": "a", "clear": true}` | nothing is on a |
| `{"stack": ["a", "b", "c"], "in": "blue_zone"}` | a tower, bottom to top; `in` is optional |

| Constraint | Meaning |
|---|---|
| `{"no_temporary": true}` | do not park objects on free slots ("don't put it on the table") |
| `{"keep": ["x"]}` | never move these objects |
| `{"never_on": [["a", "b"], ["*", "zone"]]}` | never place a on b; second entry may be an id or `slot` / `zone` / `object` |
| `{"max_stack_height": 2}` | no stack higher than this |

Constraints hold for every single placement, not only at the end. A task that
contradicts itself (a on b and b on a, something on a prism, unknown ids) is
refused with HTTP 422 and the reason.

## Requests

```
POST /v1/sessions/arm1/task
{"scene": {"scene_version": 1,
           "objects": [{"id": "red_cube_1", "type": "cube", "color": "red", "confidence": 0.96},
                       {"id": "blue_cylinder_1", "type": "cylinder", "color": "blue", "on": "red_cube_1"},
                       {"id": "green_cube_1", "type": "cube", "color": "green"}],
           "zones": [{"id": "red_zone", "color": "red", "capacity": 2}, ...],
           "free_slots": ["tmp_1"]},
 "instruction": "Put the red cube on the green cube, but don't put anything on the table."}
```

Response (shortened):

```
{"status": "ok", "plan_id": "…", "based_on_version": 1, "source": "task_astar",
 "task": {"goal": [{"object": "red_cube_1", "on": "green_cube_1"}], "constraints": {"no_temporary": true}},
 "task_description": "red_cube_1 must rest directly on green_cube_1; free slots must not be used for parking",
 "understanding": {"source": "rules", "llm_calls": 0, "rounds": 0, "examples_used": 0, "ms": 0.4},
 "steps": [{"step": 0, "action": "PICK", "object": "blue_cylinder_1", "from": "red_cube_1", "expect": {...}},
           {"step": 1, "action": "PLACE", "object": "blue_cylinder_1", "destination": "red_zone", ...}, ...],
 "info_requests": [], "llm_calls": {"understand": 0, "plan": 0},
 "timing_ms": {"understand_ms": 0.4, "parse_ms": …, "plan_ms": …, "validate_ms": …, "total_ms": 0.7}}
```

| Call | Use |
|---|---|
| `POST /v1/sessions/{sid}/task` | new task: `scene` (any form below) or `images_b64`, plus `instruction` or `task` |
| `POST /v1/sessions/{sid}/scene` | new observation (full scene or `delta`); answer says `kept` or `replanned` |
| `POST /v1/sessions/{sid}/feedback` | `{"plan_id", "step", "status", "failure"?, "scene"?}` |
| `POST /v1/perceive` | `{"images_b64": [...]}` → scene message; needs a model |
| `GET /v1/knowledge` | robot operations, rules, goal and constraint vocabulary |
| `GET /v1/metrics` | latency percentiles, how tasks were understood, model calls |
| `GET /v1/sessions/{sid}`, `GET /health` | state |

`status` is one of `ok`, `partial` (something needs another view; see
`info_requests`), `clarify` (see `question`), `infeasible` (see `reason`),
`needs_observation`, `done`, `rejected`. HTTP 409 means the message referred to an
old plan or scene version; 422 means the scene or task is inconsistent.

Scene fields for uncertainty and risk (all optional, per object): `confidence`
(identity), `pose_confidence`, `reachable`, `blocked_by: ["id", ...]` (collision
risk: this object must not be grasped until those objects have been moved),
and `color: null` for an unidentified object.

## What the scene may look like

`approaches/scene_understanding.py` reads the scene before anything else, so the robot
side does not have to produce the planner's exact format. Fields are matched by meaning:

| Input | Example |
|---|---|
| planner format | `{"objects": [{"id": "a", "type": "cube", "color": "red", "on": "b"}], "zones": [...], "free_slots": [...]}` |
| open-world / VLM | `{"objects": [{"id": "red_cylinder", "label": "red cylinder"}], "places": [...], "support": {"red_cylinder": "table"}}` |
| robot-lab export | `shape`, `position` (m), `size`, `height`, `relations` (`on_top_of`), `destinations`, `robot.held_object` |
| plain detections | `{"detections": [{"label": "red cube", "xyz": [450, 50, 20], "score": 0.93}], "zones": [{"id": "red_paper", "x": 300, "y": -210}]}` |
| images | `{"images_b64": [...], "instruction": ...}`: the model only describes the scene, the same reader checks it |

What it works out when the input does not say it:

* **what rests on what** from positions: the highest object below that overlaps in x/y
  (works whether z is the centre, bottom or top of the object, only the order matters);
* **which zone or slot** a stack stands in, from its x/y and the zone's position and size;
* colour and shape from label words; ids from the description; millimetres vs metres.

What it does **not** guess: an object reported on two supports, on an unknown id, in a
support cycle, floating above the table with nothing under it, or far off-centre on its
support. Those objects (and everything under them) get `pose_confidence: 0`, so the planner
does not touch them and the response asks for `VERIFY_SCENE`. Every inference is listed in
`scene_understanding.assumptions` and `scene_understanding.uncertain` in the response.

**Nothing has to change on the robot side.** The laptop may even give the objects new ids
in every observation: the server matches them to the previous observation (same shape and
colour first, then nearest position; the object in the gripper stays the object in the
gripper), renames them inside the remembered task, and answers with the robot's newest ids
(`scene_understanding.renamed`). If an object the task names cannot be found, the answer is
`needs_observation` with `VERIFY_SCENE`, never a guess.

If the instruction names an object by a colour the camera has not identified yet
("put the red cube ..." while one cube has `color: null`), the answer is
`needs_observation` with `REQUEST_NEW_VIEW` for the unidentified objects; the next full
scene sent to `/scene` retries the same instruction.

Measured on random simulated scenes (positions only, no `on` facts; 40 mm objects): exact up
to ±8 mm position error, never silently wrong up to ±16 mm (doubtful objects are frozen),
first silent errors at ±20 mm. Closed loop with sentences, positions-only observations
(±5 mm), grasp failures, drops, a person moving objects and hidden colours: 239/240 episodes
reached the goal over three seeds (the other stopped honestly after repeated grasp failures);
with the robot side also renumbering every object in every observation: 175/175.
Not yet measured with the real camera.

## How the task is understood (current design)

The language model understands the sentence; code never tries to cover phrasings word by
word. The rule parser is only a fallback for when no model is running or the model fails.

    sentence + objects + CURRENT arrangement + worked examples (retrieved by similarity)
      -> model writes the FINAL arrangement as goal JSON (never the moves)
      -> operations the model chose are computed exactly by code:
           {"reverse": "<id>" | "all"}   stack upside down       {"swap": ["a", "b"]}   change places
      -> checks: known ids, physically possible, not already true now (copying the present),
         constraints the sentence never mentions are dropped when they make the task impossible,
         a plan must exist; any failure goes back to the model with the reason (up to 3 rounds)
      -> vague sentence ("rearrange these cubes") -> status "clarify" with a question
      -> planner -> checked PICK/PLACE steps; replanning never calls the model again

Start: `python -m approaches.task_server --text-model ollama:llama3.1` (local text model on
any machine with Ollama), or `--model qwen3-vl-8b` on the GPU machine (images and sentences).
To make the model better at a kind of sentence, add a worked example to `SEED_EXAMPLES` in
`approaches/understand.py`, or let finished tasks be remembered (`results/task_examples.json`).

## How the task is understood (details)

```
instruction ─► cache ─► rule parser ─► language model ─► validation ─► task
                                        prompt = knowledge base        errors go back to
                                        + retrieved examples (RAG)     the model (3 rounds)
                                        + object and zone list
```

* The model is called at most once per task (plus correction rounds). Replanning
  after a failure or a scene change reuses the stored task and calls no model.
* The model sees which objects and zones exist but **not** how they are currently
  arranged, so it cannot copy the present state as the goal.
* Every answer is checked against the scene; unknown ids, contradictions and
  impossible stackings are reported back to the model, then to the user as a question.
* `--order` chooses the chain: `rules,llm` (default), `llm`, `rules`, `llm,rules`.
* Tasks that were understood and then carried out successfully are stored in
  `approaches/results/task_examples.json` and retrieved as examples for later
  instructions. Retrieval is by word overlap, no embedding model.
* The rule parser understands English only. Slovak and free phrasing need the model.

## Modes

| Mode | Task understanding | Plan | Use |
|---|---|---|---|
| `hybrid` (default) | rules, then model | symbolic planner | the recommended system |
| `vlm_plan` | rules, then model | the model writes the plan; validator checks it and sends errors back | the baseline your supervisor described |
| `symbolic` | rules and structured tasks | symbolic planner | no GPU |

The robot side sees the same endpoints and step format in all modes, so the
modes can be compared on the same scenes.

## Where each requirement is covered

| Requirement from the project description | Where | State |
|---|---|---|
| Server environment for VLM/VLA models | `approaches/task_server.py`, `approaches/llm.py` (`RepoVLM` uses `vlm.py` and `models.yaml`) | built; not yet run with a real model |
| Choosing and comparing models | `approaches/model_bench.py` | harness built; run it on the GPU machine |
| Semantic representation | scene message above; `semplan/scene.py` | built and tested |
| Interface between laptop and server | endpoints above; `GET /v1/knowledge` | built and tested |
| Generating command sequences | `semplan/task_planner.py`, `semplan/planner.py`; `vlm_plan` mode | built and tested |
| RAG / external database | `ExampleMemory` in `approaches/understand.py`; case base in `semplan/cases.py` | built and tested with mock answers |
| Knowledge base of operations and rules | `approaches/knowledge_base.json` | built; used in prompts and to check returned steps |
| Validation of plans | `semplan.scene.validate` (preconditions, task constraints, goal), `check_contract` | built and tested |
| Replanning | `semplan/service.py` | built and tested in simulation |
| Scenario A, hidden object | `color: null` / low `confidence` → `REQUEST_NEW_VIEW` | tested in simulation |
| Scenario B, stacked objects | planner | tested |
| Scenario C, collision risk | `blocked_by` | tested; the laptop has to compute the relation |
| Scenario D, not enough information | `pose_confidence`, `occluded` → `VERIFY_SCENE`, `CHANGE_VIEWPOINT` | tested in simulation |
| Latency of each stage | `timing_ms` in every response, `GET /v1/metrics` | built |

## Experiments this makes possible

1. **Models**: `model_bench` gives, per model and mode, the share of correct tasks,
   the share of plans that achieve the intended task, questions asked, model calls
   per task, latency and peak VRAM, split by kind of instruction.
2. **Hybrid against model-as-planner**: same cases, `--modes hybrid vlm_plan`.
3. **Semantic communication**: the same task with a scene message, and through
   `/v1/perceive` with images; compare bytes sent, success and latency.
4. **Effect of retrieval**: run with an empty and a filled example file.
5. **Rules first or model first**: `--order rules,llm` against `--order llm`.

Measured without a GPU (300 generated cases, `python -m approaches.model_bench`):
the rule parser alone understood 81% of the instructions correctly and asked a
question for the rest (free paraphrases and Slovak); it never produced a wrong
plan. The figures for the "mock model" rows only show that the pipeline works.

## Not done or not verified

* No real model has been run: understanding accuracy, model latency and VRAM are unknown.
* `/v1/perceive` and `RepoVLM` have never been executed against a loaded model.
* The rule parser is a set of patterns for common English phrasings; it will
  misread or reject sentences outside them (then the model, or a question, takes over).
* The general task planner is exact search; it was tested up to 5 objects with
  constraints. Larger scenes with tight constraints may hit its time budget (300 ms).
* Goals are limited to the forms in the tables. "Arrange by height" or "make a
  pyramid" are not expressible yet; the model is told to ask instead.
* The older `server.py` / `openworld.py` path is unchanged and still works separately.
