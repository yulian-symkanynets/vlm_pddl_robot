"""VLM backends: Hugging Face models on a CUDA GPU, and a mock for testing without a GPU."""
from __future__ import annotations

import copy
import json
import random
import re
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
MODELS_PATH = ROOT / 'models.yaml'


def load_registry() -> dict:
    return yaml.safe_load(MODELS_PATH.read_text())['models']


def load_images(paths: list[Path]):
    from PIL import Image, ImageOps
    images = []
    for p in paths:
        with Image.open(p) as src:
            images.append(ImageOps.exif_transpose(src).convert('RGB'))
    return images


def model_class_names(spec: dict) -> list[str]:
    """transformers Auto classes to try, in order. models.yaml: auto_class (optional).
    Newer cards (Qwen3.5, LFM2.5) use AutoModelForMultimodalLM; older ones AutoModelForImageTextToText."""
    first = {'multimodal-lm': 'AutoModelForMultimodalLM',
             'image-text-to-text': 'AutoModelForImageTextToText'}[spec.get('auto_class', 'image-text-to-text')]
    return [first] + [c for c in ('AutoModelForImageTextToText', 'AutoModelForMultimodalLM') if c != first]


def processor_kwargs(spec: dict, max_visual_tokens: int) -> dict:
    """Arguments for AutoProcessor.from_pretrained. models.yaml:
         processor_size: qwen (default)  pixel budget as Qwen-VL processors read it
                         none            no budget here; the server's image resizing limits the pixels
         processor_kwargs: {...}         passed through unchanged (e.g. LFM2.5's max_image_tokens)"""
    kw = dict(spec.get('processor_kwargs') or {})
    if spec.get('processor_size', 'qwen') == 'qwen':
        px = spec.get('patch', 32) ** 2
        # pixel budget per image; 'size' replaces the deprecated min_pixels/max_pixels kwargs
        kw['size'] = {'shortest_edge': min(256, max_visual_tokens) * px, 'longest_edge': max_visual_tokens * px}
    return kw


def template_kwargs(spec: dict) -> dict:
    """Extra chat-template options. models.yaml: chat_template_kwargs, e.g. {enable_thinking: false}
    for models that think by default (Qwen3.5-4B)."""
    return dict(spec.get('chat_template_kwargs') or {})


def weights_cached(repo: str) -> bool:
    """True if config and every weight file of the model are already in the local HF cache.
    Then the model is loaded with local_files_only: no network checks at start-up (which can
    take minutes on a network with slow DNS) and no surprise downloads."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    if not isinstance(try_to_load_from_cache(repo, 'config.json'), str):
        return False
    index = try_to_load_from_cache(repo, 'model.safetensors.index.json')
    if isinstance(index, str):
        shards = set(json.loads(Path(index).read_text())['weight_map'].values())
        return all(isinstance(try_to_load_from_cache(repo, f), str) for f in shards)
    return isinstance(try_to_load_from_cache(repo, 'model.safetensors'), str)


def download_size_gb(repo: str) -> float | None:
    """Size of the weight files on the Hub (only asked when the model still has to be downloaded)."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, files_metadata=True, timeout=15)
        return sum(f.size or 0 for f in info.siblings if f.rfilename.endswith('.safetensors')) / 1e9
    except Exception:                                    # no network, private, old library: not important
        return None


class HFVLM:
    """One loaded model. Load once, call generate() many times."""

    def __init__(self, key: str, spec: dict, quant: str = '4bit'):
        import torch
        import transformers
        from transformers import BitsAndBytesConfig
        if not torch.cuda.is_available():
            raise SystemExit('CUDA is not available. Run check_env.py; use --models mock to test without a GPU.')
        self.torch = torch
        self.key, self.spec, self.quant = key, spec, quant
        self.hf = spec['hf']
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        qc = None
        if quant == '4bit':
            qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        elif quant == '8bit':
            qc = BitsAndBytesConfig(load_in_8bit=True)
        elif quant != 'bf16':
            raise ValueError(f'unknown quant {quant}')
        self.local_only = weights_cached(self.hf)
        if self.local_only:
            print(f'{self.hf}: found in the local cache, loading without network checks', flush=True)
        else:
            gb = download_size_gb(self.hf)
            print(f'{self.hf}: NOT in the local cache. Downloading '
                  f'{f"about {gb:.1f} GB " if gb else ""}once (progress below; next starts are faster). '
                  f'To download in advance: python download_models.py {key}', flush=True)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        errors = []
        for name in model_class_names(spec):
            cls = getattr(transformers, name, None)
            if cls is None:
                errors.append(f'{name}: not in transformers {transformers.__version__}')
                continue
            try:
                try:
                    self.model = cls.from_pretrained(self.hf, dtype=dtype, device_map={'': 0}, quantization_config=qc,
                                                     attn_implementation='sdpa',
                                                     local_files_only=self.local_only).eval()
                except OSError:
                    if not self.local_only:
                        raise
                    print(f'{self.hf}: a file is missing from the cache, fetching it', flush=True)
                    self.local_only = False
                    self.model = cls.from_pretrained(self.hf, dtype=dtype, device_map={'': 0}, quantization_config=qc,
                                                     attn_implementation='sdpa').eval()
                break
            except (ValueError, KeyError) as e:          # this Auto class does not know the architecture
                errors.append(f'{name}: {e}')
        else:
            raise SystemExit(f'cannot load {self.hf}: ' + ' | '.join(errors) +
                             '\nNewer models may need a newer transformers (see the comment in models.yaml).')
        torch.cuda.synchronize()
        self.load_s = time.perf_counter() - t0
        self.weights_vram_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
        self._processors = {}
        self.revision = getattr(self.model.config, '_commit_hash', None)

    def processor(self, max_visual_tokens: int):
        if max_visual_tokens not in self._processors:
            from transformers import AutoProcessor
            kw = processor_kwargs(self.spec, max_visual_tokens)
            try:
                self._processors[max_visual_tokens] = AutoProcessor.from_pretrained(
                    self.hf, local_files_only=getattr(self, 'local_only', False), **kw)
            except OSError:                              # a processor file not cached yet: fetch it once
                self._processors[max_visual_tokens] = AutoProcessor.from_pretrained(self.hf, **kw)
        return self._processors[max_visual_tokens]

    def stream(self, image_paths: list[Path], prompt: str, max_visual_tokens: int = 1024,
               max_new_tokens: int | None = None):
        """Like generate(), but yields the answer piece by piece while the model writes it.
        Stop iterating to stop reading (the generation thread finishes its current call).
        Greedy decoding only. Not used by the benchmark; see approaches/speculative.py."""
        import threading
        from transformers import TextIteratorStreamer
        images = load_images(image_paths)
        proc = self.processor(max_visual_tokens)
        content = []
        for k in range(len(images)):
            content += [{'type': 'text', 'text': f'View {k + 1}:'}, {'type': 'image'}]
        content.append({'type': 'text', 'text': prompt})
        messages = []
        if self.spec.get('system'):
            messages.append({'role': 'system', 'content': [{'type': 'text', 'text': self.spec['system']}]})
        messages.append({'role': 'user', 'content': content})
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                        **template_kwargs(self.spec))
        inputs = proc(text=[text], images=images or None, padding=True, return_tensors='pt')
        inputs.pop('token_type_ids', None)
        inputs = inputs.to(self.model.device)
        gc = copy.deepcopy(self.model.generation_config)
        gc.num_beams, gc.use_cache = 1, True
        gc.max_new_tokens = max_new_tokens or self.spec.get('max_new_tokens', 1536)
        gc.do_sample, gc.temperature, gc.top_p, gc.top_k = False, None, None, None
        streamer = TextIteratorStreamer(proc.tokenizer, skip_prompt=True, skip_special_tokens=True)

        def work():
            with self.torch.inference_mode():
                self.model.generate(**inputs, generation_config=gc, streamer=streamer)
        threading.Thread(target=work, daemon=True).start()
        yield from streamer

    def generate(self, image_paths: list[Path], prompt: str, max_visual_tokens: int = 1024,
                 max_new_tokens: int | None = None, sample: bool = False, context=None) -> dict:
        torch = self.torch
        max_new_tokens = max_new_tokens or self.spec.get('max_new_tokens', 1536)
        t0 = time.perf_counter()
        images = load_images(image_paths)
        proc = self.processor(max_visual_tokens)
        content = []
        for k in range(len(images)):
            content += [{'type': 'text', 'text': f'View {k + 1}:'}, {'type': 'image'}]
        content.append({'type': 'text', 'text': prompt})
        messages = []
        if self.spec.get('system'):
            messages.append({'role': 'system', 'content': [{'type': 'text', 'text': self.spec['system']}]})
        messages.append({'role': 'user', 'content': content})
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                        **template_kwargs(self.spec))
        inputs = proc(text=[text], images=images or None, padding=True, return_tensors='pt')
        inputs.pop('token_type_ids', None)
        visual_tokens = None
        if 'image_grid_thw' in inputs:
            visual_tokens = int(inputs['image_grid_thw'].prod(-1).sum()) // 4   # 2x2 patch merge
        inputs = inputs.to(self.model.device)
        torch.cuda.synchronize()
        preprocess_s = time.perf_counter() - t0

        gc = copy.deepcopy(self.model.generation_config)
        gc.num_beams = 1
        gc.max_new_tokens = max_new_tokens
        gc.use_cache = True
        if sample:
            gc.do_sample, gc.temperature, gc.top_p = True, 0.6, 0.95
        else:
            gc.do_sample, gc.temperature, gc.top_p, gc.top_k = False, None, None, None
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**inputs, generation_config=gc)
        torch.cuda.synchronize()
        generate_s = time.perf_counter() - t1
        n_in = inputs['input_ids'].shape[1]
        new = out[:, n_in:]
        answer = proc.batch_decode(new, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)[0]
        n_out = int(new.shape[1])
        return {
            'text': answer,
            'input_tokens': int(n_in),
            'output_tokens': n_out,
            'visual_tokens': visual_tokens,
            'preprocess_s': round(preprocess_s, 3),
            'generate_s': round(generate_s, 3),
            'tokens_per_s': round(n_out / generate_s, 2) if generate_s else None,
            'peak_vram_gb': round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
            'truncated': n_out >= max_new_tokens,
        }


class MockVLM:
    """Answers from the ground truth (optionally with injected errors). Tests the pipeline on a Mac.

    Error injection imitates the typical VLM mistakes:
      pddl/json: a stacked object is reported as standing on the table (missed stacking)
      plan     : the first un-stacking step is dropped (picks a covered object)
    """
    load_s = 0.0
    weights_vram_gb = 0.0
    revision = 'mock'

    def __init__(self, error_rate: float = 0.0, seed: int = 0):
        self.error_rate = error_rate
        self.rng = random.Random(seed)
        self.key, self.quant, self.hf = 'mock', 'none', 'mock'

    def generate(self, image_paths, prompt, max_visual_tokens=1024, max_new_tokens=None,
                 sample=False, context=None) -> dict:
        from scenes import gt_problem
        from planning import parse_problem, solve
        scene, task, mode = context['scene'], context['task'], context['mode']
        t0 = time.perf_counter()
        corrupt = self.rng.random() < self.error_rate
        if mode == 'pddl':
            text = gt_problem(scene, task)
            m = re.search(r'\(on ([^\s()]+) ([^\s()]+)\)', text.split('(:goal')[0])
            if corrupt and m:
                text = text.replace(m.group(0), f'(on-table {m.group(1)})', 1)
            text = f'```pddl\n{text}```'
        elif mode in ('json', 'split'):
            support = dict(scene.support)
            stacked = [x for x, s in support.items() if s in scene.items]
            if corrupt and stacked:
                support[stacked[0]] = 'table'
            js = {'support': support, 'hidden': scene.hidden}
            if mode == 'json':
                js['goal'] = [list(g) for g in task.goal]
            text = json.dumps(js)
        elif mode == 'split_goal':
            goal = []
            for g in task.goal:
                if g[0] == 'in':
                    goal.append({'object': g[1], 'in': g[2]})
                elif g[0] == 'on-table':
                    goal.append({'object': g[1], 'on': 'table'})
                else:
                    goal.append({'object': g[1], 'on': g[2]})
            text = json.dumps({'goal': goal})
        else:
            _, acts, _, _ = solve(parse_problem(gt_problem(scene, task)))
            if corrupt:
                k = next((i for i, a in enumerate(acts) if a[0] == 'pick-from-item'), None)
                if k is not None:
                    acts = acts[:k] + acts[k + 2:]
            text = '```\n' + '\n'.join(f'({a} {" ".join(p)})' for a, p in acts) + '\n```'
        return {'text': text, 'input_tokens': len(prompt) // 4, 'output_tokens': len(text) // 4,
                'visual_tokens': None, 'preprocess_s': 0.0,
                'generate_s': round(time.perf_counter() - t0, 3), 'tokens_per_s': None,
                'peak_vram_gb': 0.0, 'truncated': False}


def load_model(key: str, quant: str, mock_error_rate: float = 0.0):
    if key == 'mock':
        return MockVLM(mock_error_rate)
    reg = load_registry()
    if key not in reg:
        raise SystemExit(f'unknown model {key}; known: {", ".join(reg)} or mock')
    return HFVLM(key, reg[key], quant)
