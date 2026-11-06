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


class HFVLM:
    """One loaded model. Load once, call generate() many times."""

    def __init__(self, key: str, spec: dict, quant: str = '4bit'):
        import torch
        from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
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
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.hf, dtype=dtype, device_map={'': 0}, quantization_config=qc,
            attn_implementation='sdpa').eval()
        torch.cuda.synchronize()
        self.load_s = time.perf_counter() - t0
        self.weights_vram_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
        self._processors = {}
        self.revision = getattr(self.model.config, '_commit_hash', None)

    def processor(self, max_visual_tokens: int):
        if max_visual_tokens not in self._processors:
            from transformers import AutoProcessor
            px = self.spec.get('patch', 32) ** 2
            # pixel budget per image; 'size' replaces the deprecated min_pixels/max_pixels kwargs
            size = {'shortest_edge': min(256, max_visual_tokens) * px,
                    'longest_edge': max_visual_tokens * px}
            self._processors[max_visual_tokens] = AutoProcessor.from_pretrained(self.hf, size=size)
        return self._processors[max_visual_tokens]

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
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
