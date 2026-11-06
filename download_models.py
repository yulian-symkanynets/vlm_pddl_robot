"""Download model weights into the Hugging Face cache (once, before benchmarking).

  python download_models.py                       # all models in models.yaml
  python download_models.py qwen3-vl-8b cosmos-reason1-7b
Sizes (bf16 weights on disk): 4B ~9 GB, 7-8B ~16-17 GB. Set HF_HOME to put the cache on a big disk.
"""
import sys
import time

from huggingface_hub import snapshot_download

from vlm import load_registry

reg = load_registry()
keys = sys.argv[1:] or list(reg)
for key in keys:
    if key not in reg:
        raise SystemExit(f'unknown model {key}; known: {", ".join(reg)}')
    t0 = time.time()
    print(f'--> {key}: {reg[key]["hf"]}', flush=True)
    path = snapshot_download(reg[key]['hf'], allow_patterns=['*.json', '*.safetensors', '*.txt',
                                                              '*.model', '*.jinja', '*.py', '*.tiktoken'])
    print(f'    done in {time.time() - t0:.0f} s -> {path}', flush=True)
