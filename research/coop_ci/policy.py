"""Policy interface shared by LLM and scripted cooks."""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field


def stable_seed(*parts) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)


@dataclass
class Decision:
    t: int
    agent: int
    action: str
    fields: dict = field(default_factory=dict)   # parsed reply fields other than the action
    raw: str = ""
    parse_ok: bool = True
    fallback: bool = False
    tokens: tuple = (0, 0)
    private: dict = field(default_factory=dict)  # trace-only (e.g. the LLM human's thought)


class Policy:
    role = "base"
    name = "base"
    # Attributes that are wiring, not episode state (never copied by get_state).
    _wiring = ("backend", "usage")

    def episode_start(self, env, i: int, ep_seed: int, salt: int = 0):
        self.ep_seed, self.salt = ep_seed, salt

    def decide(self, env, i: int, last_result: str, context: dict) -> Decision:
        """`context` carries runner-side information a condition may inject (e.g. the partner's
        committed macro for A3). Policies that are not allowed to see it must ignore it."""
        raise NotImplementedError

    def get_state(self) -> dict:
        return copy.deepcopy({k: v for k, v in self.__dict__.items() if k not in self._wiring})

    def set_state(self, st: dict):
        self.__dict__.update(copy.deepcopy(st))


def trace_name(episode_id: str) -> str:
    """File name (without .jsonl) of an episode's trace: '|' and '/' are not used in file names."""
    return episode_id.replace("|", "__").replace("/", "-")
