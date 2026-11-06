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
    """What the ego experienced in one episode: only events it observed (with a view radius, what
    happened within its view), plus the team score, which both cooks see."""
    seen = list(env.observed[ego])
    who = lambda a: "you" if a == ego else "partner"  # noqa: E731
    dels = [e for e in seen if e.kind == "deliver"]
    lines = [f"Episode {label} with partner {partner_id}: team score {env.score:+.0f}; deliveries you saw: "
             f"{sum(bool(e.correct) for e in dels)} correct, {sum(not e.correct for e in dels)} wrong."]
    if env.view_radius is not None:
        lines.append(f"(You only saw what happened within {env.view_radius} cells of you.)")
    adds: dict = {}          # pot -> adds seen so far for the soup in it
    carried: dict = {}       # agent -> index of the soup line it plated
    soups: list = []
    for e in seen:
        if e.kind == "pot_add":
            adds.setdefault(e.extra.get("pot", "pot"), []).append(f"{e.item.split()[-1]} ({who(e.agent)})")
        elif e.kind == "plate_soup":
            pot = e.extra.get("pot", "pot")
            soups.append(f"- soup in {pot}: {', '.join(adds.pop(pot, [])) or 'adds not seen'} -> plated by {who(e.agent)}")
            carried[e.agent] = len(soups) - 1
        elif e.kind == "deliver":
            k = carried.pop(e.agent, None)
            out = f", served by {who(e.agent)}: {'CORRECT' if e.correct else 'WRONG'}"
            if k is None:
                soups.append(f"- a soup{out}")
            else:
                soups[k] += out
    soups += [f"- unfinished in {pot}: {', '.join(a)}" for pot, a in adds.items() if a]
    lines += soups[:10]
    for a, name in ((1 - ego, "Partner (what you saw)"), (ego, "You")):
        c = Counter()
        for e in seen:
            if e.agent != a:
                continue
            if e.kind == "fetch":
                c["took " + e.item.replace("an ", "").replace("a ", "")] += 1
            elif e.kind in ("pot_add", "plate_soup", "deliver", "put_back", "counter_place"):
                c[{"pot_add": "put into a pot", "plate_soup": "plated a soup", "deliver": "delivered",
                   "put_back": "put an item back", "counter_place": "set an item on a counter"}[e.kind]] += 1
        lines.append(f"{name}: " + (", ".join(f"{k} x{v}" for k, v in c.most_common()) or "nothing") + ".")
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


REFLECT_SYS = """You keep a structured note about one cooking partner, {pid}, to cooperate better with them next time.
Rewrite the note so that it states what you now believe about this partner after the episode that just ended:
what they usually do and under which conditions, the episodes behind each belief, and your confidence between 0
and 1. Stay under {budget} characters. Reply with JSON only: {{"tendencies": [{{"claim": "...",
"conditions_of_applicability": "...", "evidence_for": ["<episode label>", ...], "evidence_against": ["<episode label>", ...],
"confidence": 0.0}}], "open_questions": ["..."]}}"""

RULES = ("standard", "static", "reflection", "evidence")
CLAIM_VERBS = ("fetch", "bring", "brought", "suppl", "add", "carr", "get", "grab", "take", "took", "focus", "prefer", "handle")


VERDICT_ADDON = (' Also state your main belief in one field, "usually_supplies": "onion", "tomato" or "unclear" '
                 '(the ingredient this partner mostly fetches for the pot).')
VERDICT_LINE = "Main belief: usually supplies "


def note_verdict(text: str) -> str:
    """Which ingredient a partner note says the partner usually supplies: 'onion', 'tomato', 'both' or
    'none'. Uses the note's explicit main belief when it has one (verdict_field), else a keyword
    heuristic over the tendency lines of a rendered note."""
    for line in text.splitlines():
        if line.startswith(VERDICT_LINE):
            v = line[len(VERDICT_LINE):].strip(" .").lower()
            return v if v in ("onion", "tomato") else "none"
    votes = Counter()
    for line in text.lower().splitlines():
        if not line.lstrip().startswith("-") or not any(v in line for v in CLAIM_VERBS):
            continue
        claim = line.split("(when:")[0]
        hits = [w for w in ("onion", "tomato") if w in claim]
        if len(hits) == 1:
            votes[hits[0]] += 1
    if not votes:
        return "none"
    top = votes.most_common()
    return top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else "both"


def render_partner_note(pid: str, note: dict) -> str:
    out = [f"Partner {pid}:"]
    if note.get("usually_supplies"):
        out.append(f"{VERDICT_LINE}{note['usually_supplies']}")
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
    ex = note.get("exceptions") or []
    if ex:
        out.append("Recorded exceptions (not yet a new habit): " + "; ".join(
            f"{e['episode']}: {e['observed']}" for e in ex[-4:]))
    return "\n".join(out)


class Memory:
    """The notes of one (stream, condition). State is plain JSON, saved after every episode."""

    def __init__(self, kind: str, budget: int = 1500, backend: Optional[Backend] = None,
                 usage: Optional[Usage] = None, max_tokens: int = 1000, rule: str = "standard", m: int = 2,
                 verdict_field: bool = False):
        """`rule` (partner notes, Exp C): standard = the B2 update; static = no updates; reflection =
        free rewrite each episode; evidence = the B2 update, but the main tendency may change only after
        `m` consecutive episodes whose proposed note contradicts it (a single deviation is kept as an
        exception)."""
        if kind not in KINDS:
            raise ValueError(f"unknown memory kind {kind}")
        if rule not in RULES:
            raise ValueError(f"unknown update rule {rule}")
        self.kind, self.budget, self.backend, self.usage, self.max_tokens = kind, budget, backend, usage, max_tokens
        self.rule, self.m = rule, m
        self.verdict_field = verdict_field  # Exp C: the note states its main belief explicitly
        self.state = {"version": 0, "general": "", "partner": {}, "raw": {}, "streak": {}}

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
        if self.rule == "static":
            return {**log, "status": "static (no update)", "version_out": st["version"]}
        if self.kind == "raw":
            st["raw"].setdefault(partner_id, []).append(summary)
            st["version"] += 1
            return {**log, "status": "ok", "version_out": st["version"]}
        if self.kind == "general":
            system = GENERAL_SYS.format(budget=self.budget)
            old = st["general"] or "(none yet)"
        else:
            system = (REFLECT_SYS if self.rule == "reflection" else PARTNER_SYS).format(pid=partner_id, budget=self.budget)
            if self.verdict_field:
                system += VERDICT_ADDON
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
            proposal = {"tendencies": js["tendencies"], "open_questions": js.get("open_questions", [])}
            if self.verdict_field:
                v = str(js.get("usually_supplies", "unclear")).strip().lower()
                proposal["usually_supplies"] = v if v in ("onion", "tomato") else "unclear"
            old_note = st["partner"].get(partner_id)
            if self.rule == "evidence" and old_note:
                gate = self._evidence_gate(partner_id, old_note, proposal, label)
                log["gate"] = gate
                if gate.startswith("held"):
                    st["version"] += 1
                    return {**log, "status": "ok (" + gate + ")", "version_out": st["version"],
                            "chars": len(self.render(partner_id))}
            if old_note and old_note.get("exceptions"):
                proposal["exceptions"] = old_note["exceptions"]
            st["partner"][partner_id] = proposal
        else:
            return {**log, "status": "parse_failed (notes kept)", "version_out": st["version"]}
        st["version"] += 1
        return {**log, "status": "ok", "version_out": st["version"], "chars": len(self.render(partner_id))}

    def _evidence_gate(self, pid: str, old_note: dict, proposal: dict, label: str) -> str:
        """Evidence-conditioned revision: if the proposal changes which ingredient the partner is said to
        supply, accept it only on the m-th consecutive contradicting episode; before that keep the old
        tendencies and record the episode as an exception."""
        v_old = note_verdict(render_partner_note(pid, old_note))
        v_new = note_verdict(render_partner_note(pid, proposal))
        streak = self.state.setdefault("streak", {})
        if v_old in ("onion", "tomato") and v_new != v_old:
            streak[pid] = streak.get(pid, 0) + 1
            if streak[pid] < self.m:
                kept = dict(old_note)
                kept["exceptions"] = (old_note.get("exceptions") or []) + [
                    {"episode": label, "observed": f"looked like {v_new}" if v_new in ("onion", "tomato") else "unclear"}]
                self.state["partner"][pid] = kept
                return f"held: {v_old} kept, contradiction {streak[pid]}/{self.m}"
            streak[pid] = 0
            return f"revised: {v_old} -> {v_new} after {self.m} contradicting episodes"
        streak[pid] = 0
        return "consistent"

    def get_state(self) -> dict:
        return copy.deepcopy(self.state)

    def set_state(self, st: dict):
        self.state = copy.deepcopy(st)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.state, f, indent=1)
