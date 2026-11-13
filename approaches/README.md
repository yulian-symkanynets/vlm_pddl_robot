# approaches

Interchangeable planning strategies for the sorting robot, behind one interface,
with one shared validator. The guides are `docs/hybrid_planning_guide.pdf` and
`docs/more_strategies.pdf` (the last six rows of the table below).

Every strategy takes the same semantic scene and returns a list of
`Move(object, destination)`. `run()` times it and replays the plan with
`semplan.scene.validate`, so nothing unvalidated reaches the robot and all
strategies are measured the same way.

```python
from approaches import run
from approaches.symbolic import GraphStrategy
from semplan.scene import parse_scene

scene = parse_scene(scene_message)            # the JSON the laptop sends
outcome = run(GraphStrategy(), scene)
outcome.status      # ok | partial | no_plan | invalid | needs_observation
outcome.moves       # [Move(obj, dst), ...]
outcome.ms          # planning time
```

## Run

```bash
pip install -r approaches/requirements.txt        # planner core itself needs nothing
python -m approaches.run_examples                 # three scenarios, all strategies, no GPU
python -m approaches.run_examples --model <key>   # real model from models.yaml (GPU machine)
python -m approaches.bench                        # writes approaches/results/bench.json
python -m pytest tests/test_approaches.py -q
python docs/make_guide.py                         # rebuilds the PDF from bench.json
python -m approaches.bench_extra && python docs/make_more_strategies.py   # second guide
STRATEGY=adaptive uvicorn approaches.serve:app --port 8020
```

## Strategies

| File | Strategy | Model at request time |
|---|---|---|
| `symbolic.py` | `GraphStrategy` constructive graph planner, A* fallback | none |
| `symbolic.py` | `PddlStrategy` scene graph to PDDL by code, pyperplan or Fast Downward | none |
| `symbolic.py` | `CaseBasedStrategy` exact structural retrieval, any fallback | none |
| `htn.py` | `HtnStrategy` small total-order HTN planner | none |
| `llm_online.py` | `LLMDirectStrategy` model writes the plan | every request |
| `llm_online.py` | `LLMModuloStrategy` model writes, validator criticises, model corrects | 1 to k calls |
| `llm_online.py` | `LLMPlusPStrategy` model translates an instruction to a goal, planner plans | only with an instruction |
| `llm_offline.py` | `synthesize_policy` + `ProgramStrategy` model writes a policy once | none |
| `llm_offline.py` | `synthesize_heuristic` + `HeuristicSearchStrategy` | none |
| `learned_policy.py` | `LearnedPolicyStrategy` trained move ranker with symbolic fallback | none (after training) |
| `active_perception.py` | `ActivePerceptionStrategy` act on what all hypotheses agree on, else look | none |
| `behavior_tree.py` | `run_tree` reactive execution with symbolic replanning | none |
| `replanning.py` | `FullReplan`, `ValidateThenReplan`, `PrefixRepair`, receding horizon | none |
| `adaptive.py` | `AdaptivePlanner` fixed order of stages with a trace | only as last fallback |
| `partial_order.py` | `deorder`, `PartialOrderStrategy` plan as a dependency graph | none |
| `cost_aware.py` | `CostAwareStrategy` fewest expected attempts, from feedback statistics | none |
| `contingent.py` | `build_tree`, `ContingentStrategy` conditional plan with observation branches | none |
| `anytime.py` | `AnytimeStrategy` valid plan at once, shorter with more time | none |
| `cpsat.py` | `CpSatStrategy` constraint model with side constraints (needs `ortools`) | none |
| `calibration.py` | `calibrate` confidence threshold chosen from a detection log | none |
| `serve.py` | FastAPI server, strategy chosen with `STRATEGY=` | |
| `task_server.py`, `task_service.py` | **the task server**: instruction or structured task in, validated plan out; see `docs/TASK_SERVER.md` | once per task |
| `understand.py` | instruction to task: rule parser, model with examples (RAG) and validation loop, cache | once per task |
| `scene_understanding.py` | any scene description (planner JSON, VLM JSON, simulator export, detections with positions, images) to the planner's scene; infers supports and zones from positions, freezes contradictions | only for images |
| `knowledge_base.json` | robot operations, rules, goal and constraint vocabulary | |
| `model_bench.py` | compare models and modes on generated instruction cases | |
| `semcom.py` | **semantic communication experiment**: image, full JSON, minimal JSON, packed binary, task-relevant subset; bytes against plan equivalence | image path only |
| `gnn_policy.py` | a trained GNN policy (PyTorch) compared with the linear ranker and the graph planner | none |

`llm.py` holds the model interface: `RepoVLM` uses this repository's models,
`MockLLM` gives scripted answers. Results obtained with `MockLLM` show the control
flow and the overhead of the surrounding code. They say nothing about the
accuracy or speed of a real model.

`semplan/` is the shared core: scene model, action model, validator, graph
planner, case base, stateful service (sessions, versions, feedback, replanning)
and a simulated robot.

## Adding a strategy

Write a class with a `name` and `propose(scene) -> Proposal`. Add it to
`make_strategy()` in `serve.py` and to `strategies()` in `bench.py`. Nothing on
the robot side changes.
