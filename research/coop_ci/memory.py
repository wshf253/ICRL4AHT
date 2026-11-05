"""Outer-loop memory for Exp B/C: text notes the ego keeps between episodes (no weight updates).

Conditions (same character budget, rendered into the ego's MEMORY block):
  none     B0  no cross-episode memory (the block says "(none yet)")
  general  B1  procedural facts and coordination lessons; the update prompt forbids partner-specific
               claims (a content control: it can still carry implicit partner information)
  partner  B2  a structured note per partner id: tendencies with conditions of applicability,
               evidence for / against (episode labels) and confidence, plus open questions
  raw      B3  the most recent episode summaries with this partner, truncated to the budget; no LLM update
  oracle   B2* the true persona label of this partner id from the first episode on, never updated: the
               ceiling for what a correct partner note can do (a privileged control, like A2 in Exp A)

Updates run after an episode only when the stream schedule allows it (held-out and return episodes
never update: strict transfer track). Every version is written to disk with the update prompt and
the raw reply. Episode summaries contain only what the ego observed (full view) and the outcome;
never the persona, its label or the partner's private thoughts.
"""
from __future__ import annotations

import copy
import json
import os
from collections import Counter
from typing import Optional

from .env import CoopEnv
from .llm import Backend, Usage, parse_json
from .policy import stable_seed

KINDS = ("none", "general", "partner", "raw", "oracle")


def episode_summary(env: CoopEnv, ego: int, partner_id: str, label: str, max_chars: int = 1400) -> str:
    """What the ego experienced in one episode, from its own perspective."""
    dels = [e for e in env.events if e.kind == "deliver"]
    lines = [f"Episode {label} with partner {partner_id}: team score {env.score:+.0f}, "
             f"{sum(e.correct for e in dels)} correct and {sum(not e.correct for e in dels)} wrong deliveries."]
    who = lambda a: "you" if a == ego else "partner"  # noqa: E731
    for s in env.soups[:8]:
        adds = ", ".join(f"{env.ing_name(k)} ({who(a)})" for _, a, k in s.adds)
        if s.correct is not None:
            out = f"served by {who(s.delivered_by)}: {'CORRECT' if s.correct else 'WRONG'}"
        elif s.plated_t is not None:
            out = "plated, not served"
        elif s.ready_t is not None:
            out = f"ready from t={s.ready_t}, never plated"
        else:
            out = "not finished"
        wait = f", waited {s.plated_t - s.ready_t} steps for a plate" if s.plated_t and s.ready_t else ""
        lines.append(f"- soup {s.idx + 1} ({s.pot}, order of adding): {adds} -> {out}{wait}.")
    for a, name in ((1 - ego, "Partner"), (ego, "You")):
        c = Counter()
        for e in env.events:
            if e.agent != a:
                continue
            if e.kind == "fetch":
                c["took " + e.item.replace("an ", "").replace("a ", "")] += 1
            elif e.kind in ("pot_add", "plate_soup", "deliver", "put_back", "counter_place"):
                c[{"pot_add": "put into a pot", "plate_soup": "plated a soup", "deliver": "delivered",
                   "put_back": "put an item back", "counter_place": "set an item on a counter"}[e.kind]] += 1
        lines.append(f"{name}: " + (", ".join(f"{k} x{v}" for k, v in c.most_common()) or "did nothing") + ".")
    return "\n".join(lines)[:max_chars]


GENERAL_SYS = """You keep short notes that help you cook better in future Overcooked episodes with ANY partner.
Write procedural facts and coordination lessons: how the kitchen works, what wasted time, what to do in which
situation. Do NOT write anything about a particular partner (no partner ids, habits or preferences of a specific
partner). Keep what is still useful from your old notes, fix what the new episode contradicts, stay under {budget}
characters. Reply with JSON only: {{"notes": "<your updated notes>"}}"""

PARTNER_SYS = """You keep a structured note about one cooking partner, {pid}, to cooperate better with them next time.
Record tendencies you have evidence for: what they usually do and under which conditions. For each tendency give
the episodes that support it and any that contradict it, and a confidence between 0 and 1. A single deviation is an
exception, not a new habit; change a tendency only when the evidence against it keeps coming. Stay under {budget}
characters. Reply with JSON only: {{"tendencies": [{{"claim": "...", "conditions_of_applicability": "...",
"evidence_for": ["<episode label>", ...], "evidence_against": ["<episode label>", ...], "confidence": 0.0}}],
"open_questions": ["..."]}}"""


def render_partner_note(pid: str, note: dict) -> str:
    out = [f"Partner {pid}:"]
    for t in note.get("tendencies", [])[:8]:
        if not isinstance(t, dict):
            continue
        ev = ",".join(map(str, t.get("evidence_for", []) or [])) or "-"
        ag = ",".join(map(str, t.get("evidence_against", []) or [])) or "-"
        out.append(f"- {t.get('claim', '')} (when: {t.get('conditions_of_applicability', '-')}; "
                   f"for: {ev}; against: {ag}; confidence {t.get('confidence', '?')})")
    qs = [str(q) for q in note.get("open_questions", []) or []][:4]
    if qs:
        out.append("Open questions: " + "; ".join(qs))
    return "\n".join(out)


class Memory:
    """The notes of one (stream, condition). State is plain JSON, saved after every episode."""

    def __init__(self, kind: str, budget: int = 1500, backend: Optional[Backend] = None,
                 usage: Optional[Usage] = None, max_tokens: int = 1000):
        if kind not in KINDS:
            raise ValueError(f"unknown memory kind {kind}")
        self.kind, self.budget, self.backend, self.usage, self.max_tokens = kind, budget, backend, usage, max_tokens
        self.state = {"version": 0, "general": "", "partner": {}, "raw": {}}

    def render(self, partner_id: str) -> str:
        st = self.state
        if self.kind == "general":
            text = st["general"]
        elif self.kind == "partner":
            note = st["partner"].get(partner_id)
            text = render_partner_note(partner_id, note) if note else ""
        elif self.kind == "raw":
            text, kept = "", []
            for summ in reversed(st["raw"].get(partner_id, [])):  # newest first, as many as fit
                if len("\n\n".join([summ] + kept)) > self.budget:
                    break
                kept.insert(0, summ)
            text = "\n\n".join(kept)
        else:
            text = ""
        return text[:self.budget]

    def update(self, summary: str, partner_id: str, label: str, seed_parts: tuple) -> dict:
        """Fold one episode into the notes. Returns a log record (also when nothing changes)."""
        st = self.state
        log = {"kind": self.kind, "partner_id": partner_id, "episode": label, "version_in": st["version"]}
        if self.kind in ("none", "oracle"):
            return {**log, "status": "no_memory" if self.kind == "none" else "oracle (fixed)"}
        if self.kind == "raw":
            st["raw"].setdefault(partner_id, []).append(summary)
            st["version"] += 1
            return {**log, "status": "ok", "version_out": st["version"]}
        if self.kind == "general":
            system = GENERAL_SYS.format(budget=self.budget)
            old = st["general"] or "(none yet)"
        else:
            system = PARTNER_SYS.format(pid=partner_id, budget=self.budget)
            note = st["partner"].get(partner_id)
            old = json.dumps(note) if note else "(none yet)"
        user = f"YOUR CURRENT NOTES:\n{old}\n\nWHAT HAPPENED IN THE EPISODE THAT JUST ENDED:\n{summary}"
        r = self.backend.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                              max_tokens=self.max_tokens, seed=stable_seed("memory", *seed_parts))
        if self.usage is not None:
            self.usage.add(r)
        js = parse_json(r.text) or {}
        log.update(prompt=user, raw=r.text[:4000], tokens=(r.prompt_tokens, r.completion_tokens))
        if self.kind == "general" and isinstance(js.get("notes"), str) and js["notes"].strip():
            st["general"] = js["notes"].strip()[:self.budget]
        elif self.kind == "partner" and isinstance(js.get("tendencies"), list):
            st["partner"][partner_id] = {"tendencies": js["tendencies"], "open_questions": js.get("open_questions", [])}
        else:
            return {**log, "status": "parse_failed (notes kept)", "version_out": st["version"]}
        st["version"] += 1
        return {**log, "status": "ok", "version_out": st["version"], "chars": len(self.render(partner_id))}

    def get_state(self) -> dict:
        return copy.deepcopy(self.state)

    def set_state(self, st: dict):
        self.state = copy.deepcopy(st)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.state, f, indent=1)
