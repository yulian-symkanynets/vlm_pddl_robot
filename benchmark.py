"""Benchmark VLMs on image -> PDDL (and baselines), with timing and validation.

Examples
  # pipeline test without GPU (answers from ground truth, 30 % injected errors)
  python benchmark.py --models mock --mock-error-rate 0.3

  # one model, PDDL mode only, quick
  python benchmark.py --models qwen3-vl-8b --modes pddl --scenes s5_reorder --limit 3

  # full comparison
  python benchmark.py --models qwen3-vl-4b qwen3-vl-8b cosmos-reason1-7b --modes pddl json plan

  # image resolution sweep (semantic communication: fewer visual tokens = less data)
  python benchmark.py --models qwen3-vl-8b --modes json --max-visual-tokens 256 512 1024 2048

Output: results/<timestamp>/runs.jsonl (one line per run), raw/ (prompt + answer + PDDL), config.json
Then:   python summarize.py results/<timestamp>
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

from evaluate import evaluate, ground_truth
from prompts import MODES, build_prompt
from scenes import ROOT, load_config
from vlm import load_model


def versions() -> dict:
    out = {'python': platform.python_version(), 'host': platform.node()}
    for mod in ('torch', 'transformers', 'bitsandbytes', 'unified_planning'):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    try:
        import torch
        if torch.cuda.is_available():
            out['gpu'] = torch.cuda.get_device_name(0)
            out['cuda'] = torch.version.cuda
    except Exception:
        pass
    return out


def jobs(cfg, args):
    for scene in cfg.scenes:
        if args.scenes != ['all'] and scene.id not in args.scenes:
            continue
        view_sets = scene.view_sets if args.view_sets == 'all' else scene.view_sets[:1]
        for task in scene.tasks:
            if args.tasks != ['all'] and task.task_id not in args.tasks:
                continue
            for views in view_sets:
                for mode in args.modes:
                    for mvt in args.max_visual_tokens:
                        for rep in range(args.repeats):
                            yield scene, task, views, mode, mvt, rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+', default=['mock'], help='keys from models.yaml, or mock')
    ap.add_argument('--modes', nargs='+', default=list(MODES), choices=MODES)
    ap.add_argument('--scenes', nargs='+', default=['all'])
    ap.add_argument('--tasks', nargs='+', default=['all'])
    ap.add_argument('--view-sets', choices=['all', 'first'], default='all')
    ap.add_argument('--max-visual-tokens', nargs='+', type=int, default=[1024],
                    help='per image; several values = resolution sweep')
    ap.add_argument('--quant', choices=['4bit', '8bit', 'bf16'], default='4bit')
    ap.add_argument('--max-new-tokens', type=int, default=None, help='override models.yaml')
    ap.add_argument('--sample', action='store_true', help='sampling (T=0.6, top_p=0.95) instead of greedy')
    ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--warmup', action='store_true', help='one untimed generation after loading')
    ap.add_argument('--no-hints', action='store_true', help='do not give object IDs to the model')
    ap.add_argument('--planner', default='auto', help='auto | fast-downward | fast-downward-opt | pyperplan')
    ap.add_argument('--limit', type=int, default=None, help='max runs per model (quick test)')
    ap.add_argument('--mock-error-rate', type=float, default=0.0)
    ap.add_argument('--dry-run', action='store_true', help='print the first prompt of each mode and exit')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    cfg = load_config()
    all_jobs = list(jobs(cfg, args))
    if not all_jobs:
        raise SystemExit('no jobs: check --scenes / --tasks')

    if args.dry_run:
        seen = set()
        for scene, task, views, mode, mvt, rep in all_jobs:
            if mode not in seen:
                seen.add(mode)
                print(f'===== {mode} | {scene.id} {task.task_id} {views}\n')
                print(build_prompt(mode, scene, task, len(views), not args.no_hints), '\n')
        print(f'{len(all_jobs)} runs per model')
        return

    out = Path(args.out) if args.out else ROOT / 'results' / datetime.now().strftime('%Y%m%d_%H%M%S')
    (out / 'raw').mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps({'args': vars(args), 'versions': versions()}, indent=2))
    runs_path = out / 'runs.jsonl'
    print(f'{len(all_jobs)} runs per model -> {runs_path}')

    # ground truth once (also checks that every gt problem is solvable)
    for scene, task, *_ in all_jobs:
        gt = ground_truth(scene, task, args.planner)
        if gt['plan_len'] is None:
            raise SystemExit(f'ground truth {scene.id}/{task.task_id} has no plan: fix scenes.yaml')

    for key in args.models:
        print(f'\n=== loading {key} ({args.quant})', flush=True)
        model = load_model(key, args.quant, args.mock_error_rate)
        print(f'loaded in {model.load_s:.1f} s, weights {model.weights_vram_gb:.1f} GB', flush=True)
        model_jobs = all_jobs[:args.limit] if args.limit else all_jobs
        if args.warmup and key != 'mock':
            s, t, v, m, mvt, _ = model_jobs[0]
            model.generate([s.views[x] for x in v], build_prompt(m, s, t, len(v), not args.no_hints),
                           mvt, 32, context={'scene': s, 'task': t, 'mode': m})

        for n, (scene, task, views, mode, mvt, rep) in enumerate(model_jobs, 1):
            image_paths = [scene.views[v] for v in views]
            prompt = build_prompt(mode, scene, task, len(views), not args.no_hints)
            run_id = f'{key}_{scene.id}_{task.task_id}_{"+".join(views)}_{mode}_v{mvt}_r{rep}'
            t0 = time.perf_counter()
            try:
                gen = model.generate(image_paths, prompt, mvt, args.max_new_tokens, args.sample,
                                     context={'scene': scene, 'task': task, 'mode': mode})
            except Exception as e:          # e.g. CUDA OOM: record and continue
                gen = {'text': '', 'error': f'{type(e).__name__}: {e}'}
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            ev = evaluate(mode, gen['text'], scene, task, args.planner) if gen['text'] else \
                {'stage': 'generate', 'success': False, 'error': gen.get('error')}
            total_s = time.perf_counter() - t0

            raw = out / 'raw' / f'{run_id}.txt'
            raw.write_text(f'### PROMPT\n{prompt}\n\n### ANSWER\n{gen["text"]}\n\n'
                           f'### PROBLEM USED\n{ev.get("problem_text") or "-"}\n\n'
                           f'### PLAN\n{json.dumps(ev.get("plan"))}\n')
            rec = {
                'run_id': run_id, 'model': key, 'hf': model.hf, 'quant': model.quant,
                'mode': mode, 'scene': scene.id, 'task': task.task_id, 'views': '+'.join(views),
                'n_images': len(views), 'image_bytes': sum(p.stat().st_size for p in image_paths),
                'max_visual_tokens': mvt, 'hints': not args.no_hints, 'repeat': rep,
                'load_s': round(model.load_s, 2),
                **{k: v for k, v in gen.items() if k != 'text'},
                **{k: v for k, v in ev.items() if k != 'problem_text'},
                'total_s': round(total_s, 3), 'prompt_chars': len(prompt),
            }
            with runs_path.open('a') as f:
                f.write(json.dumps(rec) + '\n')
            print(f'[{n}/{len(model_jobs)}] {run_id}: {"OK " if ev["success"] else "FAIL"} '
                  f'stage={ev["stage"]} gen={gen.get("generate_s", 0):.1f}s '
                  f'{("- " + str(ev["error"])[:90]) if ev.get("error") else ""}', flush=True)
        del model
        try:
            import gc, torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
    print(f'\nDone. Summary: python summarize.py {out}')


if __name__ == '__main__':
    main()
