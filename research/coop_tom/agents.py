"""Policies: LLM agent (the robot), LLM-simulated human, scripted partner, scripted reference cook."""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Optional

from .env import CoopEnv, decode_ingredients, is_cooked, is_plate, multiset_contains, multiset_minus
from .llm import Backend, Usage, parse_json
from .planner import parse_macro, pot_side_cell, reachable_targets, valid_actions
from .prompts import agent_system_prompt, decision_user_prompt, human_system_prompt

MAX_TOKENS = {"direct": 64, "human": 120}


def from_counter(env: CoopEnv, i: int, ingredient: int) -> Optional[str]:
    """'pick up from counter x y' for a visible, reachable counter holding this ingredient.
    Reusing set-aside ingredients keeps the few counters from filling up."""
    held = [c for c in env.counters if env.in_view(i, c) and env.cell(c)[1]
            and not is_plate(env.cell(c)[1]) and decode_ingredients(env.cell(c)[1]) == [ingredient]]
    reach = reachable_targets(env, i, held)
    return f"pick up from counter {reach[0][0]} {reach[0][1]}" if reach else None


def stable_seed(*parts) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)


@dataclass
class Decision:
    t: int
    agent: int
    action: str
    fields: dict = field(default_factory=dict)
    raw: str = ""
    parse_ok: bool = True
    fallback: bool = False
    tokens: tuple = (0, 0)


class Policy:
    role = "base"

    def episode_start(self, env: CoopEnv, i: int, ep_seed: int):
        self.ep_seed = ep_seed

    def decide(self, env: CoopEnv, i: int, last_result: str) -> Decision:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# LLM policies
# --------------------------------------------------------------------------- #
class _LLMPolicyBase(Policy):
    def __init__(self, backend: Backend, usage: Usage):
        self.backend, self.usage = backend, usage

    def _ask(self, env, i, system, user, max_tokens, seed) -> Decision:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        r = self.backend.chat(msgs, max_tokens=max_tokens, seed=seed)
        self.usage.add(r)
        js = parse_json(r.text)
        act = str(js.get("action", "")) if js else ""
        m = parse_macro(act, env) if act else None
        tokens = (r.prompt_tokens, r.completion_tokens)
        if m is None:  # one repair round
            msgs += [{"role": "assistant", "content": r.text or "(empty)"},
                     {"role": "user", "content": f"That was not a valid reply. Use exactly one action from the ACTIONS list "
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
        return Decision(env.t, i, str(m), js, r.text, tokens=tokens)


class LLMAgentPolicy(_LLMPolicyBase):
    """The robot. `condition` selects the output block; oracles inject ground truth."""

    role = "agent"

    def __init__(self, backend, usage, condition="direct", reasoning_words=80, oracle_recipe=False,
                 oracle_persona_text="", partner_id="P1", n_events=12):
        super().__init__(backend, usage)
        self.condition, self.reasoning_words = condition, reasoning_words
        self.oracle_recipe, self.oracle_persona_text = oracle_recipe, oracle_persona_text
        self.partner_id, self.n_events = partner_id, n_events

    def system_prompt(self, env, i) -> str:
        # Memory notes (Exp 3-6) are not part of this first version.
        return agent_system_prompt(env, i, self.condition, self.reasoning_words, self.partner_id,
                                   self.oracle_persona_text)

    def user_prompt(self, env, i, last_result) -> str:
        return decision_user_prompt(env.observation_text(i, self.oracle_recipe, self.n_events), valid_actions(env, i), last_result)

    def decide(self, env, i, last_result):
        mt = MAX_TOKENS["direct"] if self.condition == "direct" else int(self.reasoning_words * 2.5) + 200
        if self.condition in ("tom_slots", "tom_2nd"):
            mt = 400
        return self._ask(env, i, self.system_prompt(env, i), self.user_prompt(env, i, last_result), mt,
                         stable_seed(self.ep_seed, env.t, i, "act"))


class LLMHumanPolicy(_LLMPolicyBase):
    role = "partner"

    def __init__(self, backend, usage, persona_text: str):
        super().__init__(backend, usage)
        self.persona_text = persona_text

    def decide(self, env, i, last_result):
        sys = human_system_prompt(env, i, self.persona_text)
        user = decision_user_prompt(env.observation_text(i, False), valid_actions(env, i), last_result)
        return self._ask(env, i, sys, user, MAX_TOKENS["human"], stable_seed(self.ep_seed, env.t, i, "human"))


# --------------------------------------------------------------------------- #
# Scripted partner (reproducibility control, X3) - implements the same personas by rule
# --------------------------------------------------------------------------- #
NEW_ORDER_WAIT = 10


class ScriptedPartner(Policy):
    role = "partner"

    def __init__(self, spec: dict):
        self.spec = spec

    def episode_start(self, env, i, ep_seed):
        super().episode_start(env, i, ep_seed)
        self.rng = random.Random(ep_seed)
        self.known_order: Optional[list] = None
        self.pending_pause = 0
        self.seen_plate_t: Optional[int] = None   # last plating of a soup this partner saw
        self.order_at_plate: Optional[list] = None

    def _order(self, env, i):
        o = env.visible_order(i)
        if o is not None:
            self.known_order = list(o)
        return self.known_order

    def _look(self, env, i) -> str:
        targets = env.indicators or env.buttons
        if not targets:
            return "wait 3"
        t = targets[0]
        from .planner import bfs
        adj = {c for c in env.neighbors(t) if env._walkable(c)}
        path = bfs(env, env.agent_pos(i), adj, set())
        if path:
            return f"move to {path[-1][0]} {path[-1][1]}"
        return "wait 3"

    def decide(self, env, i, last_result):
        kind = self.spec.get("kind", "demo")
        act = self._demo(env, i) if kind == "demo" else self._signal(env, i)
        return Decision(env.t, i, act)

    def _demo(self, env, i) -> str:
        order = self._order(env, i)
        inv = env.inventory(i)
        pot = env.pots[0]
        status, contents, _ = env.pot_status(pot)
        soup = env._open_soup.get(pot)
        own_adds = sum(1 for (_, a, _) in soup.adds if a == i) if soup else 0
        if self.pending_pause > 0:
            n, self.pending_pause = self.pending_pause, 0
            return f"wait {n}"
        if order is None:
            return self._look(env, i)
        # The order display changes only when the plated soup is delivered. After seeing a
        # soup plated, wait for the new order (up to NEW_ORDER_WAIT steps, in case the next
        # order equals the old one) instead of starting the next soup for the old order.
        plates = [e for e in env.observed[i] if e.kind == "plate_soup"]
        if plates and plates[-1].t != self.seen_plate_t:
            self.seen_plate_t, self.order_at_plate = plates[-1].t, list(order)
        if (self.seen_plate_t is not None and env.t - self.seen_plate_t < NEW_ORDER_WAIT
                and order == self.order_at_plate and not contents):
            return "wait 1"
        if inv != 0 and not is_plate(inv) and not is_cooked(inv):
            # Walk to the pot first, then decide while looking at it: the order display
            # and the pot are both in view there, so a changed order or a partner's add
            # since the fetch is noticed (the persona never adds an off-order ingredient).
            side = pot_side_cell(env, i, pot)
            if side is not None and env.agent_pos(i) != side:
                return f"move to {side[0]} {side[1]}"
            held = decode_ingredients(inv)[0]
            if status == "idle" and len(contents) < 3 and multiset_contains(order, contents + [held]):
                self.pending_pause = int(self.spec.get("pause", 0))
                return "put in pot"
            return "place on counter" if status == "idle" else "wait 2"
        if inv != 0:
            return "place on counter"
        if status != "idle" or len(contents) >= 3 or not multiset_contains(order, contents):
            return "wait 3"
        if own_adds >= int(self.spec.get("n_add", 3)):
            return "wait 3"
        remaining = multiset_minus(order, contents)
        reach = {k for k in env.piles if reachable_targets(env, i, env.piles[k])}
        remaining = [r for r in remaining if r in reach]
        if not remaining:
            return "wait 3"
        rule = self.spec.get("order", "any")
        if rule == "minority_first":
            remaining.sort(key=lambda k: (order.count(k), k))
        elif rule == "majority_first":
            remaining.sort(key=lambda k: (-order.count(k), k))
        elif rule == "random":
            self.rng.shuffle(remaining)
        return from_counter(env, i, remaining[0]) or f"fetch {env.ing_name(remaining[0])}"

    def _signal(self, env, i) -> str:
        """Test-Time Protocol personas: encode the order by where a signal item sits."""
        order = self._order(env, i)
        inv = env.inventory(i)
        if order is None:
            return self._look(env, i)
        conv = {k: tuple(v) for k, v in self.spec["convention"].items()}
        want = conv.get(env.recipe_str(order))
        sig = int(self.spec.get("signal_ingredient", 2))
        wrong = [c for c in conv.values() if c != want and env.cell(c)[1] != 0]
        if inv != 0:
            if want and env.cell(want)[1] == 0:
                return f"place on counter {want[0]} {want[1]}"
            free = [c for c in env.counters if env.cell(c)[1] == 0 and c not in conv.values()]
            return f"place on counter {free[0][0]} {free[0][1]}" if free else "wait 2"
        if wrong:
            return f"pick up from counter {wrong[0][0]} {wrong[0][1]}"
        if want and env.cell(want)[1] == 0:
            return f"fetch {env.ing_name(sig)}"
        return "wait 3"


# --------------------------------------------------------------------------- #
# Scripted reference cook (non-LLM baseline for the robot slot)
# --------------------------------------------------------------------------- #
class ScriptedCook(Policy):
    """Uses only what the robot can observe. Waits for evidence while the consistent set
    has >1 recipe, commits once it is a singleton or the partner has been idle `patience` steps."""

    role = "agent"

    def __init__(self, patience: int = 30, oracle_recipe: bool = False):
        self.patience, self.oracle_recipe = patience, oracle_recipe

    def _candidates(self, env, contents):
        return [env.current_recipe] if self.oracle_recipe else env.consistent_recipes(contents)

    def decide(self, env, i, last_result):
        inv = env.inventory(i)
        pot = env.pots[0]
        status, contents, _ = env.pot_status(pot)
        if is_cooked(inv):
            if self.oracle_recipe and sorted(decode_ingredients(inv)) != sorted(env.current_recipe):
                return Decision(env.t, i, "place on counter")  # knows it is wrong: do not serve it
            return Decision(env.t, i, "deliver")
        if inv == 1:
            return Decision(env.t, i, "plate soup" if status in ("cooking", "ready") else "wait 2")
        if inv != 0:
            side = pot_side_cell(env, i, pot)
            if side is not None and env.agent_pos(i) != side:
                return Decision(env.t, i, f"move to {side[0]} {side[1]}")  # look at the pot before adding
            held = decode_ingredients(inv)[0]
            fits = any(multiset_contains(r, contents + [held]) for r in self._candidates(env, contents))
            ok = status == "idle" and len(contents) < 3 and fits
            return Decision(env.t, i, "put in pot" if ok else "place on counter")
        if status in ("cooking", "ready"):
            return Decision(env.t, i, "fetch plate")
        # Patience runs from the start of the current soup: the partner's last pot add,
        # or the last delivery the robot made or saw (a new order starts there).
        marks = [e.t for e in env.observed[i] if (e.agent != i and e.kind == "pot_add") or e.kind == "deliver"]
        idle_for = env.t - (marks[-1] if marks else 0)
        cands = self._candidates(env, contents)
        if not self.oracle_recipe:
            if not contents and idle_for < self.patience:
                return Decision(env.t, i, "wait 2")  # let the informed partner start the soup
        if not cands:
            return Decision(env.t, i, "wait 3")
        if len(cands) > 1 and idle_for < self.patience:
            return Decision(env.t, i, "wait 2")
        target = cands[0]
        remaining = multiset_minus(target, contents)
        reach = [k for k in remaining if k in env.piles and reachable_targets(env, i, env.piles[k])]
        if not reach:
            return Decision(env.t, i, "wait 3")
        return Decision(env.t, i, from_counter(env, i, reach[0]) or f"fetch {env.ing_name(reach[0])}")
