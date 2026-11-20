# Live benchmark (run on HEKATE)

Real requests against the running server, with graphs. It measures the real model, the real
network path and the real timings, and judges every answer independently of the server.

```bash
# 1. on HEKATE (WSL), server without the experience database, so times are model times
NO_MEMORY=1 ./serve.sh start qwen3-vl-4b 4bit

# 2. quick look (a few minutes), then everything
python live_bench.py --quick --label qwen3-vl-4b-4bit
python live_bench.py --label qwen3-vl-4b-4bit

# 3. your own photos (see live_photos/README.md), and a run from your laptop over the network
python live_bench.py --suites photo --photos live_photos
python live_bench.py --url http://192.168.73.15:8010 --label via-lan

# 4. several models one after another, compared side by side
scripts/live_compare.sh qwen3-vl-2b qwen3-vl-4b qwen3-vl-8b
```

Result folder `results/live/<time>_<label>/`: `report.html` (one file, graphs inside, send it as is),
`results.json` (all numbers), `rows.csv`, `charts/*.png`. Rebuild a report or compare runs with
`python live_report.py results/live/<run> [<other run> ...]`.

## What is tested

| Suite | Input | Question it answers |
|---|---|---|
| `json` | sentence + scene JSON | Does the model turn the sentence into the right goal? Per kind of task: `examples`, `constraints`, `sort`, `stack` (4 to 32 objects), `novel` (tasks nobody wrote code for), `language` (Slovak, Czech, German), `refusal` (impossible or unclear: a question is the right answer) |
| `scale` | scene + goal, no model | How do planning time and plan length grow with the number of objects? |
| `photo` | photo with a known true scene | `perceive`: what is read right at every resolution, and how long it takes. `photo_plan`: the same task from the photo and from the JSON (seconds, bytes sent, success) |

## How a request is judged

The plan is replayed on the TRUE scene with the benchmark's own rules (`live_score.py`: hand empty,
object free, target free, room in the zone, lid open). Then a reference written by a person must be
true on the final arrangement (a goal formula that the model never sees). Outcomes:

- **correct**: the plan works and does the task (for `refusal` cases a question or `no_plan` is correct)
- **wrong plan**: the server said `ok`, the plan does not do the task. The dangerous kind; the goal is 0
- **asked** / **no plan**: the model asked a question or no plan exists, but a plan was expected
- **unexpected ok**: a plan came back where a refusal was right
- **error**: HTTP or server error. **partial** (perceive only): the scene was read, but not exactly

## Reading the report

`outcomes` (what came back per kind of task), `latency` (perception / goal / planning / rest),
`scaling` (time and moves against objects), `photo_resolution` (seconds, tokens and accuracy per level),
`photo_vs_json` (the same task by both routes), `quality` (proven shortest or only verified),
and with several runs `compare`. Below the charts: the failures, each with the goal the model wrote in
plain words, and the table of all requests.

## Notes

- With the experience database on (default), repeated inputs are answered from it and look fast. Use
  `NO_MEMORY=1` (or `--clear-memory`). The report warns when it sees cache hits.
- The first request of a GPU model is slower (warm-up); the runner sends one unrecorded request first.
- Cases live in `live_cases.py`; add a dict to `json_cases()` (a sentence, a scene, a reference formula).
- Without a GPU the same run works against `python server.py --model mock` (what the tests do): the
  stand-in model cannot read free wording or photos, so most `novel`, `language` and `photo` cases fail.
  That is a check of the benchmark, not a result.
- Not covered: hardware, real arm motion, cameras. Everything here is server side.
