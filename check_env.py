"""Check the environment: GPU, libraries, planners, model cache. Run this first on a new machine."""
import importlib
import sys

ok = True


def show(name, good, info=''):
    global ok
    ok &= bool(good)
    print(f'[{"OK" if good else "!!"}] {name:28s} {info}')


print('python', sys.version.split()[0])
for mod in ('yaml', 'PIL', 'unified_planning', 'torch', 'transformers', 'accelerate', 'bitsandbytes'):
    try:
        m = importlib.import_module(mod)
        show(mod, True, getattr(m, '__version__', ''))
    except Exception as e:
        show(mod, mod in ('torch', 'transformers', 'accelerate', 'bitsandbytes') and False, f'missing ({e.__class__.__name__})')

try:
    import torch
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        show('CUDA', True, f'{torch.cuda.get_device_name(0)}, {free / 2**30:.1f}/{total / 2**30:.1f} GB free, '
                           f'CUDA {torch.version.cuda}, bf16={torch.cuda.is_bf16_supported()}')
        x = torch.ones(3, device='cuda')
        show('CUDA compute', (x + x).sum().item() == 6)
    else:
        show('CUDA', False, 'not available -> only "--models mock" and validate_pddl.py will work')
except ImportError:
    show('CUDA', False, 'torch not installed (fine on the Mac for mock/validation)')

try:
    import transformers
    from packaging.version import Version
    show('transformers >= 4.57 (Qwen3-VL)', Version(transformers.__version__) >= Version('4.57.0'),
         transformers.__version__)
except Exception:
    pass

from scenes import load_config, gt_problem          # noqa: E402
from planning import parse_problem, solve, simulate  # noqa: E402

cfg = load_config()
s = cfg.scene('s5_reorder')
p = parse_problem(gt_problem(s, s.tasks[0]))
for planner in ('fast-downward', 'fast-downward-opt', 'pyperplan'):
    try:
        status, acts, dt, _ = solve(p, planner)
        show(f'planner {planner}', acts and simulate(p, acts)['ok'], f'{len(acts)} steps, {dt:.2f} s')
    except Exception as e:
        show(f'planner {planner}', False, f'{type(e).__name__}: {str(e)[:80]}')

try:
    from huggingface_hub import scan_cache_dir
    import yaml
    from vlm import load_registry
    cached = {r.repo_id for r in scan_cache_dir().repos}
    for key, spec in load_registry().items():
        print(f'     model {key:22s} {"cached" if spec["hf"] in cached else "not downloaded"}')
except Exception as e:
    print('     (model cache not checked:', type(e).__name__, ')')

print('\nALL OK' if ok else '\nSome checks failed (see !! lines).')
