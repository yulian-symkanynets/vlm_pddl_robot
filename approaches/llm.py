"""Language-model access for the LLM-assisted strategies.

LLM is the only interface the strategies use: complete(prompt) -> text.

  RepoVLM   the models of this repository (vlm.py, models.yaml), text-only prompts.
            Needs the GPU machine.
  MockLLM   scripted answers, for testing the control flow on a machine without a GPU.

A mock says nothing about a real model's accuracy or speed. Numbers measured
with MockLLM only show the overhead of the surrounding code.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Protocol


class LLM(Protocol):
    calls: int

    def complete(self, prompt: str) -> str: ...


class MockLLM:
    """Answers from a function of the prompt, or from a fixed list of answers in order."""

    def __init__(self, responder: Callable[[str], str] | list[str]):
        self.responder = responder
        self.calls = 0
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        if callable(self.responder):
            return self.responder(prompt)
        return self.responder[min(self.calls - 1, len(self.responder) - 1)]


class RepoVLM:
    """Adapter around this repository's model loader (text-only use).

        from approaches.llm import RepoVLM
        llm = RepoVLM("qwen3vl-8b", quant="4bit")     # a key from models.yaml
    """

    def __init__(self, key: str, quant: str = "4bit", max_new_tokens: int = 512):
        from vlm import load_model
        self.model = load_model(key, quant)
        self.max_new_tokens = max_new_tokens
        self.calls = 0
        self.generate_s = 0.0                       # accumulated generation time

    def complete(self, prompt: str) -> str:
        self.calls += 1
        out = self.model.generate([], prompt, max_new_tokens=self.max_new_tokens)
        self.generate_s += out.get("generate_s", 0.0)
        return out["text"]


# ---------------------------------------------------------------- parsing model output

def strip_reasoning(text: str) -> str:
    return text.rsplit("</think>", 1)[-1].strip()


def extract_json(text: str):
    """First balanced {...} or [...] in the text, parsed. None if there is none."""
    text = strip_reasoning(text)
    for open_c, close_c in ("{}", "[]"):
        i = text.find(open_c)
        while i >= 0:
            depth = 0
            for j in range(i, len(text)):
                depth += text[j] == open_c
                depth -= text[j] == close_c
                if depth == 0:
                    try:
                        return json.loads(text[i:j + 1])
                    except json.JSONDecodeError:
                        break
            i = text.find(open_c, i + 1)
    return None


def extract_code(text: str) -> str:
    """Contents of the first ``` block, or the whole text if there is no block."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", strip_reasoning(text), re.S)
    return (m.group(1) if m else strip_reasoning(text)).strip()


def extract_moves(text: str) -> list[tuple[str, str]]:
    """Lines of the form  MOVE <object> <destination>  (brackets and arrows tolerated)."""
    out = []
    for line in strip_reasoning(text).splitlines():
        m = re.search(r"MOVE[\s(]+([\w-]+)[\s,>\-]+([\w-]+)", line, re.I)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


class OllamaLLM:
    """A local text model served by Ollama (http://127.0.0.1:11434), e.g. llama3.1 or qwen2.5.

        llm = OllamaLLM("llama3.1")

    Runs on the machine itself (no data leaves it). Used for understanding instructions;
    it never plans moves.
    """

    def __init__(self, model: str = "llama3.1", url: str = "http://127.0.0.1:11434",
                 temperature: float = 0.0, timeout_s: float = 120.0, json_mode: bool = True):
        self.key, self.url, self.temperature = model, url.rstrip("/"), temperature
        self.timeout_s, self.json_mode = timeout_s, json_mode
        self.calls = 0
        self.generate_s = 0.0

    def complete(self, prompt: str) -> str:
        import time
        import urllib.request
        body = {"model": self.key, "prompt": prompt, "stream": False,
                "options": {"temperature": self.temperature}}
        if self.json_mode:
            body["format"] = "json"
        req = urllib.request.Request(self.url + "/api/generate", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            out = json.load(r)
        self.calls += 1
        self.generate_s += time.perf_counter() - t0
        return out.get("response", "")
