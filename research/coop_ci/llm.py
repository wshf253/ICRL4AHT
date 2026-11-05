"""LLM backends over the OpenAI-compatible Chat Completions API (stdlib only).

The same code talks to a local vLLM server (stage 1: Qwen3.8-27B) and later to a
frontier API (stage 2): swapping models means pointing a backend name at another
entry in configs/backends.yaml. No openai/vllm/torch dependency in this env.
"""
from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S)


@dataclass
class LLMResult:
    text: str
    reasoning: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    error: str = ""
    finish_reason: str = ""

    @property
    def thinking(self) -> bool:
        """The server produced reasoning text despite thinking being requested off.
        An empty <think></think> block (some templates emit one when thinking is off) is not."""
        return bool(re.sub(r"</?think>", "", self.reasoning).strip())


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: int = 0
    thinking: int = 0
    truncated: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, r: LLMResult):
        with self.lock:
            self.calls += 1
            self.prompt_tokens += r.prompt_tokens
            self.completion_tokens += r.completion_tokens
            self.errors += bool(r.error)
            self.thinking += r.thinking
            self.truncated += r.finish_reason == "length"

    def as_dict(self):
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "errors": self.errors,
                "thinking": self.thinking, "truncated": self.truncated}


class Backend:
    name = "base"

    def chat(self, messages: list[dict], max_tokens: Optional[int] = None, seed: Optional[int] = None,
             json_mode: Optional[bool] = None) -> LLMResult:
        raise NotImplementedError


class OpenAICompatBackend(Backend):
    """POST /chat/completions. Thinking control goes in `chat_template_kwargs` (vLLM)."""

    def __init__(self, name: str, base_url: str, model: str, api_key_env: Optional[str] = None,
                 params: Optional[dict] = None, extra_body: Optional[dict] = None, json_mode: bool = True,
                 timeout_s: float = 120.0, max_retries: int = 4, send_seed: bool = True):
        self.name, self.model = name, model
        self.base_url = base_url.rstrip("/")
        self.key = os.environ.get(api_key_env, "") if api_key_env else ""
        self.params = dict(params or {})
        self.extra_body = dict(extra_body or {})
        self.json_mode, self.timeout_s = json_mode, timeout_s
        self.max_retries, self.send_seed = max_retries, send_seed
        # Cluster loopback/node addresses must not go through HTTP proxies.
        self.opener = build_opener(ProxyHandler({})) if _is_local(self.base_url) else build_opener()

    def payload(self, messages, max_tokens=None, seed=None, json_mode=None) -> dict:
        body: dict[str, Any] = {"model": self.model, "messages": messages, **self.params}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if seed is not None and self.send_seed:
            body["seed"] = int(seed) % (2**31)
        if self.json_mode if json_mode is None else json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(self.extra_body)  # raw REST: chat_template_kwargs, top_k, ... are top-level
        return body

    def post(self, endpoint: str, body: Optional[dict] = None) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        req = Request(f"{self.base_url}/{endpoint}", headers=headers,
                      data=None if body is None else json.dumps(body).encode())
        with self.opener.open(req, timeout=self.timeout_s) as resp:
            return json.load(resp)

    def chat(self, messages, max_tokens=None, seed=None, json_mode=None) -> LLMResult:
        body = self.payload(messages, max_tokens, seed, json_mode)
        last_err = ""
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                resp = self.post("chat/completions", body)
                choice = resp["choices"][0]
                msg = choice.get("message") or {}
                text = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                m = _THINK_RE.search(text)
                if m:  # no reasoning parser on the server: strip inline thinking
                    reasoning = reasoning or m.group(0)
                    text = _THINK_RE.sub("", text).strip()
                u = resp.get("usage") or {}
                return LLMResult(text, reasoning, u.get("prompt_tokens", 0) or 0, u.get("completion_tokens", 0) or 0,
                                 time.time() - t0, finish_reason=choice.get("finish_reason") or "")
            except HTTPError as e:  # 4xx are config errors: do not retry
                last_err = f"HTTP {e.code}: {e.read(500).decode(errors='replace')}"
                if 400 <= e.code < 500 and e.code != 429:
                    break
            except (URLError, TimeoutError, ValueError, KeyError, IndexError) as e:
                last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(30.0, 2.0 ** attempt + random.random()))
        return LLMResult("", error=last_err)

    def models(self) -> list[str]:
        return [m["id"] for m in self.post("models").get("data", [])]


def _is_local(url: str) -> bool:
    host = re.sub(r"^https?://", "", url).split("/")[0].split(":")[0]
    return host in ("localhost", "127.0.0.1") or host.startswith(("10.", "192.168.", "172."))


class MockBackend(Backend):
    """Offline stand-in for pipeline tests: well-formed JSON for every prompt type, no ToM.

    It reads the `VALID_ACTIONS:` and `MENU:` lines that the prompts always include and
    prefers task progress. Every call is recorded (tests check probes stay forked)."""

    def __init__(self, name="mock", seed: int = 0):
        self.name = name
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self.calls: list[list[dict]] = []

    def chat(self, messages, max_tokens=None, seed=None, json_mode=None) -> LLMResult:
        with self._lock:
            self.calls.append(messages)
        rng = random.Random(seed) if seed is not None else self._rng
        text = "\n".join(m["content"] for m in messages)
        last = messages[-1]["content"]
        menu = _grab_list(text, "MENU:")
        if "PROBE" in last:
            out = {"recipe_probs": {r: round(1 / max(1, len(menu)), 3) for r in menu},
                   "partner_type": "A", "partner_next_action": "wait 2"}
        else:
            valid = [a for a in _grab_list(last, "VALID_ACTIONS:") if "<" not in a]
            pri = ["deliver", "plate soup", "put in", "fetch plate"]
            act = next((a for p in pri for a in valid if a.startswith(p)), None)
            if act is None:
                fetches = [a for a in valid if a.startswith("fetch")]
                act = rng.choice(fetches + ["wait 2"]) if fetches else "wait 2"
            if '"notes"' in text and "YOUR CURRENT NOTES" in last:   # memory update (Exp B)
                return LLMResult(json.dumps({"notes": "mock lesson: plate soups as soon as they are ready"}),
                                 finish_reason="stop")
            if '"tendencies"' in text and "YOUR CURRENT NOTES" in last:
                return LLMResult(json.dumps({"tendencies": [{"claim": "mock claim", "conditions_of_applicability": "always",
                                                             "evidence_for": ["E1"], "evidence_against": [],
                                                             "confidence": 0.5}], "open_questions": []}),
                                 finish_reason="stop")
            out = {}
            for k in ("thought", "plan", "partner_current_subgoal", "evidence", "partner_tendency",
                      "my_complementary_action_reason"):
                if f'"{k}"' in text:
                    out[k] = "mock"
            if '"reasoning"' in text:
                out["reasoning"] = "mock"
            if '"partner_reasoning"' in text:
                out["partner_reasoning"] = "mock"
            if '"recipe_belief"' in text:
                out.update({"recipe_belief": {r: round(1 / max(1, len(menu)), 3) for r in menu},
                            "partner_type": "unknown", "partner_next_action": "wait 2"})
            if '"partner_expects_me_to"' in text:
                out["partner_expects_me_to"] = "wait 2"
            out["action"] = act
        s = json.dumps(out)
        return LLMResult(s, prompt_tokens=len(text) // 4, completion_tokens=len(s) // 4, finish_reason="stop")


def _grab_list(text: str, tag: str) -> list[str]:
    for line in reversed(text.splitlines()):
        if line.strip().startswith(tag):
            return [x.strip() for x in line.split(tag, 1)[1].split("|") if x.strip()]
    return []


def build_backend(name: str, spec: dict) -> Backend:
    kind = spec.get("type", "openai_compat")
    if kind == "mock":
        return MockBackend(name, seed=spec.get("seed", 0))
    if kind == "openai_compat":
        base_url = os.environ.get(spec["base_url_env"], "") if spec.get("base_url_env") else ""
        base_url = base_url or spec.get("base_url", "")
        if not base_url or "SET_ME" in base_url:
            raise ValueError(f"backend '{name}': set base_url (or the env var {spec.get('base_url_env')})")
        extra = dict(spec.get("extra_body") or {})
        env_kwargs = os.environ.get(spec["chat_template_kwargs_env"], "") if spec.get("chat_template_kwargs_env") else ""
        if env_kwargs:  # e.g. set by the serving script after its thinking-off check
            extra["chat_template_kwargs"] = json.loads(env_kwargs)
        return OpenAICompatBackend(
            name=name, base_url=base_url, model=os.environ.get(spec.get("model_env", ""), "") or spec["model"],
            api_key_env=spec.get("api_key_env"), params=spec.get("params"), extra_body=extra,
            json_mode=spec.get("json_mode", True), timeout_s=spec.get("timeout_s", 120.0),
            max_retries=spec.get("max_retries", 4), send_seed=spec.get("send_seed", True),
        )
    raise ValueError(f"unknown backend type {kind}")


def parse_json(text: str) -> Optional[dict]:
    """Best-effort JSON object extraction (handles code fences / leading prose)."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        v = json.loads(t)
        return v if isinstance(v, dict) else None
    except Exception:
        pass
    depth, start = 0, None
    for k, ch in enumerate(t):
        if ch == "{":
            if depth == 0:
                start = k
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    v = json.loads(t[start:k + 1])
                    if isinstance(v, dict):
                        return v
                except Exception:
                    start = None
    return None
