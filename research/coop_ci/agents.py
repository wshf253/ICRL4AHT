"""LLM cooks: the ego (conditions A0-A4) and the LLM-simulated human partner.

Both can run on the same served model (stage 1: Qwen3.8-27B); they differ in their prompts.
Neither keeps state across decisions besides the episode seed: given the env and executor
state, a decision depends only on the prompt and the request seed (tested for the partner).
"""
from __future__ import annotations

from .env import CoopEnv
from .llm import Backend, Usage, parse_json
from .planner import parse_macro, valid_actions
from .policy import Decision, Policy, stable_seed
from .prompts import OUTPUT_OF, decision_user_prompt, ego_system_prompt, human_system_prompt

MAX_TOKENS = {"A0": 96, "human": 160}


class _LLMPolicyBase(Policy):
    def __init__(self, backend: Backend, usage: Usage):
        self.backend, self.usage = backend, usage

    def _ask(self, env: CoopEnv, i: int, system: str, user: str, max_tokens: int, seed: int) -> Decision:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        r = self.backend.chat(msgs, max_tokens=max_tokens, seed=seed)
        self.usage.add(r)
        js = parse_json(r.text)
        act = str(js.get("action", "")) if js else ""
        m = parse_macro(act, env) if act else None
        tokens = (r.prompt_tokens, r.completion_tokens)
        if m is not None:
            return Decision(env.t, i, str(m), js, r.text, tokens=tokens)
        # One repair round (the same policy for every condition).
        msgs += [{"role": "assistant", "content": r.text or "(empty)"},
                 {"role": "user", "content": "That was not a valid reply. Use exactly one action from the ACTIONS list "
                                             f"(e.g. one of: {', '.join(valid_actions(env, i)[:6])}) in the JSON format required."}]
        r2 = self.backend.chat(msgs, max_tokens=max_tokens, seed=seed + 1)
        self.usage.add(r2)
        js2 = parse_json(r2.text)
        act2 = str(js2.get("action", "")) if js2 else ""
        m = parse_macro(act2, env) if act2 else None
        tokens = (tokens[0] + r2.prompt_tokens, tokens[1] + r2.completion_tokens)
        if m is not None:
            return Decision(env.t, i, str(m), js2 or {}, r2.text, parse_ok=False, tokens=tokens)
        return Decision(env.t, i, "wait 1", js or {}, r.text, parse_ok=False, fallback=True, tokens=tokens)


class LLMEgoPolicy(_LLMPolicyBase):
    """The ego cook. `condition` in A0 A1 A1c A2 A3 A4; `profile_label` is the injected profile
    (true persona for A2, a mismatched one for A4); A3 reads the partner's committed macro from
    the runner context."""

    role = "ego"

    def __init__(self, backend, usage, condition="A0", words=80, partner_id="P-0000", profile_label="",
                 n_events=10):
        super().__init__(backend, usage)
        if condition not in OUTPUT_OF:
            raise ValueError(f"unknown condition {condition}")
        if condition in ("A2", "A4") and not profile_label:
            raise ValueError(f"{condition} needs a profile label")
        self.condition, self.words, self.partner_id = condition, words, partner_id
        self.profile_label = profile_label if condition in ("A2", "A4") else ""
        self.n_events = n_events
        self.name = condition

    def max_tokens(self) -> int:
        # A1 and A1c share one cap (token-matched control).
        return MAX_TOKENS["A0"] if OUTPUT_OF[self.condition] == "A0" else int(self.words * 2.5) + 120

    def system_prompt(self, env) -> str:
        return ego_system_prompt(env, self.condition, self.words, self.partner_id, self.profile_label)

    def user_prompt(self, env, i, last_result, context=None) -> str:
        injected = ""
        if self.condition == "A3":
            injected = f"Partner is currently doing: {(context or {}).get('partner_doing') or 'nothing yet'}"
        return decision_user_prompt(env.observation_text(i, self.n_events), valid_actions(env, i), last_result, injected)

    def decide(self, env, i, last_result, context=None):
        return self._ask(env, i, self.system_prompt(env), self.user_prompt(env, i, last_result, context),
                         self.max_tokens(), stable_seed(self.ep_seed, self.salt, env.t, i, "ego"))


class LLMHumanPolicy(_LLMPolicyBase):
    """LLM-simulated human with a fixed persona and no memory. Its thought stays in the trace."""

    role = "partner"

    def __init__(self, backend, usage, persona_text: str, n_events=10):
        super().__init__(backend, usage)
        self.persona_text, self.n_events = persona_text, n_events
        self.name = "llm_human"

    def prompts(self, env, i, last_result):
        return (human_system_prompt(env, self.persona_text),
                decision_user_prompt(env.observation_text(i, self.n_events), valid_actions(env, i), last_result))

    def decide(self, env, i, last_result, context=None):
        sys, user = self.prompts(env, i, last_result)
        d = self._ask(env, i, sys, user, MAX_TOKENS["human"], stable_seed(self.ep_seed, self.salt, env.t, i, "human"))
        d.private["thought"] = str(d.fields.pop("thought", ""))
        return d
