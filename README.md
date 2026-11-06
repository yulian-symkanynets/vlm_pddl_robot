# vlm_pddl: image → PDDL benchmark

Test which VLM turns photos of the table plus a task instruction into a **correct PDDL problem**,
how long it takes, and whether the resulting plan would actually work in the real scene.

The **domain** (`domain/tabletop.pddl`, the robot's skills) is written once. For every image and
task, only the **problem** is generated, either by the VLM or by Python from the VLM's JSON.

## Three modes (compared side by side)

| Mode | What the VLM writes | What Python does |
|---|---|---|
| `pddl` | the whole PDDL problem (objects, :init, :goal) | parse → check → plan → run the plan on the ground truth |
| `json` | only `{"support": …, "hidden": …, "goal": …}` | generate the PDDL problem → same checks |
| `plan` | the action sequence directly (baseline M1, no planner) | run the plan on the ground truth |

## What is validated for each answer

```
extract  → is there a (define (problem …)) / JSON / action list in the answer?
parse    → does it parse against the domain? (syntax, types, unknown predicates or objects)
lint     → is the state physically consistent? (one support per object, clear/hand-empty correct, slots)
compare  → support facts vs ground truth (F1, exact), goal facts vs ground truth
solve    → does a planner (Fast Downward / pyperplan) find a plan for the VLM's problem?
execute  → does THAT plan work on the GROUND-TRUTH problem, step by step, and reach the true goal?
```

**`success` = the plan works in the real scene.** A problem can parse and solve and still fail
here, because the model saw the scene wrongly (e.g. it missed a stack) or misunderstood the goal.

Timing per run: model load, preprocessing, generation, tokens in/out, tokens/s, visual tokens,
peak VRAM, parse, planner, and end-to-end time. JPEG bytes are logged too (for the semantic-communication comparison).

## Setup

**GPU machine (Linux + NVIDIA, e.g. the RTX A4500):**
```bash
./setup.sh                                  # .venv, torch+CUDA, transformers, bitsandbytes, planners
source .venv/bin/activate
python check_env.py                         # GPU, versions, planners, which models are cached
python download_models.py qwen3-vl-8b       # or no argument = all models in models.yaml
python -m pytest tests -q                   # 26 offline tests, no GPU needed
```
If `nvidia-smi` shows an old driver, set another torch build: `CUDA_INDEX=https://download.pytorch.org/whl/cu121 ./setup.sh`.

**Mac (no GPU):** `./setup.sh --cpu`. Validation, tests and `--models mock` work there.

## Run

```bash
# 1. pipeline check without a model (answers from ground truth, 30 % injected errors)
python benchmark.py --models mock --mock-error-rate 0.3

# 2. look at the prompts
python benchmark.py --dry-run

# 3. quick real test: one model, one scene
python benchmark.py --models qwen3-vl-8b --scenes s5_reorder --tasks swap_stacks --modes pddl json plan

# 4. full comparison
python benchmark.py --models qwen3-vl-4b qwen3-vl-8b qwen2.5-vl-7b cosmos-reason1-7b --warmup

# 5. resolution sweep (less image data → faster, but is it still correct?)
python benchmark.py --models qwen3-vl-8b --modes json --max-visual-tokens 256 512 1024 2048

# 6. effect of views (scene 5 has close / top / wide / close+top / close+higher+top)
python benchmark.py --models qwen3-vl-8b --scenes s5_reorder

# summary table (success %, parse %, perception F1, times, VRAM, most common failures)
python summarize.py results/<timestamp>
python summarize.py results/<timestamp> --by model mode views
python summarize.py results/<timestamp> --by model max_visual_tokens
```

Useful flags: `--quant 4bit|8bit|bf16`, `--no-hints` (model must find and name the objects itself),
`--repeats 3 --sample` (variance), `--limit 5` (quick test), `--planner fast-downward-opt`.

Every run's prompt, raw answer, PDDL and plan are in `results/<timestamp>/raw/`.

## Validate by hand

```bash
python validate_pddl.py --scene s5_reorder --task swap_stacks --print-gt     # correct problem + optimal plan
python validate_pddl.py answer.txt                                            # parse, consistency, plan
python validate_pddl.py answer.txt --scene s5_reorder --task swap_stacks      # + does it work in reality
python validate_pddl.py --plan plan.txt --scene s5_reorder --task swap_stacks # check a VLM-written plan
```
You can paste a full model answer (with ```pddl fences or <think> reasoning) into the file; the
problem or plan is extracted automatically.

## Adding scenes, tasks, places, models

- **New photo / scene:** put the image in `images/` and add an entry in `scenes/scenes.yaml` with the
  ground-truth `support` (what each object rests on) and one or more `tasks`. The ground-truth
  problem is generated automatically; `pytest` checks that it is solvable.
- **New task on an existing photo:** add `{id, instruction, goal}` under `tasks`. `goal: sort` means
  "every object onto its color zone".
- **Box / shelf:** add a place under `env.places` (or per scene `env: {places: …}`):
  `box_1: {kind: box, slots: 3, open: false}` (closed lid → plan starts with `open-lid`),
  `shelf_low: {kind: shelf, slots: 2, clearance_mm: 60}` (taller items get no `fits` fact → cannot go there).
  No PDDL editing is needed for either.
- **New robot skill:** add an action to `domain/tabletop.pddl` (the only hand-written PDDL).
- **New model:** add an entry to `models.yaml` (any model that works with `AutoModelForImageTextToText`).

## Files

| File | Purpose |
|---|---|
| `domain/tabletop.pddl` | fixed robot domain (STRIPS + typing): pick/place on table, object, place slot; open-lid; observe |
| `scenes/scenes.yaml` | lab places, objects, scenes, views, ground truth, tasks |
| `scenes.py` | config loading, automatic PDDL problem generation |
| `prompts.py` | prompts for `pddl`, `json`, `plan` |
| `vlm.py` | Hugging Face VLM runner with timing/VRAM; `MockVLM` for tests |
| `models.yaml` | model registry: Qwen3-VL 4B/8B(+Thinking), Qwen2.5-VL-7B, Cosmos-Reason1-7B |
| `planning.py` | extraction, parsing (unified-planning), consistency lint, planners, step-by-step simulator |
| `evaluate.py` | scores one answer against the ground truth |
| `benchmark.py` | runs models × scenes × tasks × views × modes × resolutions, writes `runs.jsonl` |
| `summarize.py` | summary tables (`summary.md`, `summary.csv`) |
| `validate_pddl.py` | manual validation of a problem or plan |
| `check_env.py`, `download_models.py`, `setup.sh` | setup |
| `tests/test_offline.py` | 26 tests: ground truth solvable, missed stack detected, bottom-block pick rejected, box/shelf cases |

## Notes and limits

- **Check the ground-truth labels.** `scenes.yaml` was labelled from the phone photos in `cosmo/images`.
  Check each `support` entry before trusting the scores (the `s5_reorder` wide view might also show the labelled cube).
- The photos are phone photos at an angle. The robot's D435 looks almost straight down. Add D435
  frames (and an external side view) as new views of the same scenes to measure what the robot will really get.
- The planner may use a free paper as a temporary spot (e.g. it puts the red cylinder on blue_zone
  for a moment). That is valid in the domain. Add a `temp` place if you want explicit temporary spots.
- The domain has no geometry (collisions, reach, stability) and no camera-collision rule yet (scenario C).
  Those checks belong to David's local side or a later domain version.
- `success` means "logically correct for the real scene", not "verified safe to execute".
