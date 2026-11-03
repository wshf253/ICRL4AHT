"""Macro-actions on top of OvercookedV2 primitive actions.

LLMs choose macro-actions ("fetch onion", "put in pot", ...); a BFS executor turns
them into primitive moves. The LLM is queried only when a macro finishes or fails,
which cuts calls from ~1 per step to tens per episode.
"""
from __future__ import annotations

import random
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .env import A_INTERACT, A_STAY, DIR_VEC, VEC_TO_ACTION, CoopEnv, is_cooked, is_plate, decode_ingredients

MAX_WAIT = 10
DEFAULT_TIMEOUT = 40

GRAMMAR = """\
- fetch <ingredient>          take one ingredient from its pile (hands must be empty)
- fetch plate                 take a plate from a plate pile (hands must be empty)
- put in pot                  put the held ingredient into the pot
- plate soup                  with a plate in hand, take the finished soup from the pot (waits by the pot if still cooking)
- deliver                     bring the held dish to the serving window X
- place on counter <x> <y>    put the held item on that counter (omit x y for the nearest free one)
- pick up from counter <x> <y>
- press button                press the order button L (costs 5 points; shows the order for 10 steps)  [button layouts]
- move to <x> <y>             walk to a floor cell (e.g. to look at something)
- wait <n>                    stay put for n steps (1-10)"""


@dataclass
class Macro:
    kind: str
    arg: Optional[str] = None
    xy: Optional[tuple[int, int]] = None
    n: int = 1
    raw: str = ""

    def __str__(self):
        return self.raw or self.kind


def grammar_for(env: CoopEnv) -> str:
    """Action list for this layout (drops the button line when there is no button)."""
    lines = GRAMMAR.splitlines()
    if not env.buttons:
        lines = [l for l in lines if "press button" not in l]
    return "\n".join(l.replace("  [button layouts]", "") for l in lines)


def parse_macro(text: str, env: CoopEnv) -> Optional[Macro]:
    s = text.strip().lower().strip(".").replace("_", " ")
    s = re.sub(r"[(),]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    m = re.fullmatch(r"fetch (\w+)", s)
    if m:
        name = m.group(1)
        if name in ("plate", "dish", "bowl"):
            return Macro("fetch_plate", raw="fetch plate")
        names = [n.lower() for n in env.ingredient_names]
        if name in names and names.index(name) in env.piles:
            return Macro("fetch", arg=names.index(name), raw=f"fetch {name}")
        return None
    if re.fullmatch(r"put (it )?in(to)? (the )?pot", s):
        return Macro("put_in_pot", raw="put in pot")
    if re.fullmatch(r"plate (the )?soup", s):
        return Macro("plate_soup", raw="plate soup")
    if re.fullmatch(r"deliver( dish)?", s):
        return Macro("deliver", raw="deliver")
    m = re.fullmatch(r"place on counter(?: (\d+) (\d+))?", s)
    if m:
        xy = (int(m.group(1)), int(m.group(2))) if m.group(1) else None
        return Macro("place_counter", xy=xy, raw=s)
    m = re.fullmatch(r"pick up from counter (\d+) (\d+)", s)
    if m:
        return Macro("pick_counter", xy=(int(m.group(1)), int(m.group(2))), raw=s)
    if re.fullmatch(r"press (the )?button", s):
        return Macro("press_button", raw="press button")
    m = re.fullmatch(r"move to (\d+) (\d+)", s)
    if m:
        return Macro("move_to", xy=(int(m.group(1)), int(m.group(2))), raw=s)
    m = re.fullmatch(r"wait(?: (\d+))?(?: steps?)?", s)
    if m:
        n = int(m.group(1)) if m.group(1) else 1
        return Macro("wait", n=max(1, min(MAX_WAIT, n)), raw=f"wait {max(1, min(MAX_WAIT, n))}")
    return None


def bfs(env: CoopEnv, start, goals: set, blocked: set) -> Optional[list]:
    """Shortest walkable path from start to any cell in goals (path includes start)."""
    if start in goals:
        return [start]
    q, prev = deque([start]), {start: None}
    while q:
        c = q.popleft()
        for n in env.neighbors(c):
            if n in prev or not env._walkable(n) or n in blocked:
                continue
            prev[n] = c
            if n in goals:
                path = [n]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return path[::-1]
            q.append(n)
    return None


def reachable_targets(env: CoopEnv, i: int, targets: list) -> list:
    """Targets (non-walkable cells) that agent i can stand next to; sorted by path length."""
    me = env.agent_pos(i)
    other = {env.agent_pos(j) for j in range(env.num_agents) if j != i}
    out = []
    for t in targets:
        adj = {n for n in env.neighbors(t) if env._walkable(n)}
        p = bfs(env, me, adj, blocked=set()) if adj else None
        if p is not None:
            p2 = bfs(env, me, adj, blocked=other)
            out.append((len(p2) if p2 else len(p) + 5, t))
    return [t for _, t in sorted(out)]


@dataclass
class MacroState:
    macro: Macro
    started_t: int
    timeout: int = DEFAULT_TIMEOUT
    steps: int = 0
    status: str = "running"  # running | done | failed
    reason: str = ""
    target: Optional[tuple] = None
    inv0: int = 0
    pots_at_start: dict = field(default_factory=dict)  # pot -> contents when the macro was chosen
    blocked: int = 0
    trace: list = field(default_factory=list)


class Executor:
    """Turns the current macro of agent i into one primitive action per step."""

    def __init__(self, env: CoopEnv, i: int, salt: int = 0):
        self.env, self.i, self.salt = env, i, salt
        self.ms: Optional[MacroState] = None
        self.intended: Optional[tuple] = None  # next cell this agent wants to enter (for yielding)
        self._moved_from: Optional[tuple] = None  # cell we tried to leave last step (collision detection)

    def _rng(self) -> random.Random:
        return random.Random(self.salt * 1000003 + self.env.t * 31 + self.i)

    def sidestep(self, avoid: set) -> Optional[int]:
        """Move to a random free neighbouring floor cell (deadlock breaking / yielding)."""
        env, me = self.env, self.env.agent_pos(self.i)
        occ = {env.agent_pos(j) for j in range(env.num_agents) if j != self.i} | set(avoid)
        cand = [n for n in env.neighbors(me) if env._walkable(n) and n not in occ]
        if not cand:
            return None
        n = self._rng().choice(sorted(cand))
        return VEC_TO_ACTION[(n[0] - me[0], n[1] - me[1])]

    def start(self, macro: Macro):
        self.ms = MacroState(macro, self.env.t, inv0=self.env.inventory(self.i),
                             pots_at_start={p: self.env.pot_status(p)[1] for p in self.env.pots})
        err = self._precheck(macro)
        if err:
            self.ms.status, self.ms.reason = "failed", err

    def idle(self) -> bool:
        return self.ms is None or self.ms.status != "running"

    # ---- validity ---- #
    def _precheck(self, m: Macro) -> str:
        env, i = self.env, self.i
        inv = env.inventory(i)
        if m.kind in ("fetch", "fetch_plate", "pick_counter", "press_button") and inv != 0:
            return f"hands are not empty (holding {env.item_text(inv)})"
        if m.kind == "put_in_pot" and (inv == 0 or is_plate(inv)):
            return "not holding an ingredient"
        if m.kind == "plate_soup" and inv != 1:
            return "not holding an empty plate"
        if m.kind == "deliver" and not is_cooked(inv):
            return "not holding a dish"
        if m.kind == "place_counter" and inv == 0:
            return "holding nothing"
        if m.kind == "move_to" and (m.xy is None or not (0 <= m.xy[0] < env.W and 0 <= m.xy[1] < env.H) or not env._walkable(m.xy)):
            return f"{m.xy} is not a floor cell"
        targets = self._targets(m)
        if m.kind not in ("wait", "move_to") and not targets:
            return "no reachable target for this action"
        return ""

    def _targets(self, m: Macro) -> list:
        env, i = self.env, self.i
        if m.kind == "fetch":
            cand = env.piles.get(m.arg, [])
        elif m.kind == "fetch_plate":
            cand = env.plate_piles
        elif m.kind == "put_in_pot":
            cand = [p for p in env.pots if env.pot_status(p)[0] == "idle" and len(env.pot_status(p)[1]) < 3]
        elif m.kind == "plate_soup":
            cand = [p for p in env.pots if env.pot_status(p)[0] in ("ready", "cooking")]
        elif m.kind == "deliver":
            cand = env.goals
        elif m.kind == "place_counter":
            free = [c for c in env.counters if env.cell(c)[1] == 0]
            cand = [m.xy] if m.xy is not None else free
            cand = [c for c in cand if c in free]
        elif m.kind == "pick_counter":
            cand = [m.xy] if (m.xy in env.counters and env.cell(m.xy)[1] != 0) else []
        elif m.kind == "press_button":
            cand = env.buttons
        else:
            return []
        return reachable_targets(env, i, cand)

    # ---- one step ---- #
    def act(self) -> int:
        self.intended = None
        tried_from, self._moved_from = self._moved_from, None
        ms, env, i = self.ms, self.env, self.i
        if ms is None or ms.status != "running":
            return A_STAY
        ms.steps += 1
        m = ms.macro
        if m.kind == "wait":
            if ms.steps >= m.n:
                ms.status = "done"
            return A_STAY
        if ms.steps > ms.timeout:
            ms.status, ms.reason = "failed", "timed out"
            return A_STAY
        me = env.agent_pos(i)
        other = {env.agent_pos(j) for j in range(env.num_agents) if j != i}
        if m.kind == "move_to":
            if me == m.xy:
                ms.status = "done"
                return A_STAY
            path = bfs(env, me, {m.xy}, other) or bfs(env, me, {m.xy}, set())
            return self._step_along(path, tried_from)
        if self._complete():
            ms.status = "done"
            return A_STAY
        targets = self._targets(m)
        if not targets:
            ms.status, ms.reason = "failed", "target no longer available"
            return A_STAY
        t = targets[0]
        ms.target = t
        adj = {n for n in env.neighbors(t) if env._walkable(n)}
        if me in adj:
            need = (t[0] - me[0], t[1] - me[1])
            if DIR_VEC[env.agent_dir(i)] != need:
                return VEC_TO_ACTION[need]  # blocked move = turn to face
            if m.kind == "put_in_pot" and env.pot_status(t)[1] != ms.pots_at_start.get(t):
                # The pot changed while walking (e.g. the partner added an ingredient):
                # hand control back so the agent decides with what it now sees. Same rule
                # for every agent; it needs no knowledge of the order.
                now = ", ".join(env.ing_name(c) for c in env.pot_status(t)[1]) or "empty"
                ms.status, ms.reason = "failed", f"the pot changed while you were walking (now: {now})"
                return A_STAY
            if m.kind == "plate_soup" and env.pot_status(t)[0] == "cooking":
                return A_STAY  # wait by the pot
            if m.kind == "press_button":
                ms.reason = "pressed"
            return A_INTERACT
        path = bfs(env, me, adj, other) or bfs(env, me, adj, set())
        return self._step_along(path, tried_from)

    def _step_along(self, path, tried_from=None) -> int:
        if not path or len(path) < 2:
            return A_STAY
        a, b = path[0], path[1]
        others = {self.env.agent_pos(j) for j in range(self.env.num_agents) if j != self.i}
        # Blocked = partner stands on the next cell, or last step's move from here did not happen
        # (both agents stepped into the same cell). Break the symmetry with a random sidestep.
        if b in others or tried_from == a:
            self.ms.blocked += 1
            if self.ms.blocked >= 2 and self._rng().random() < 0.5:
                self.ms.blocked = 0
                side = self.sidestep(avoid=set())
                if side is not None:
                    return side
        else:
            self.ms.blocked = 0
        self.intended = b
        self._moved_from = a
        return VEC_TO_ACTION[(b[0] - a[0], b[1] - a[1])]

    def _complete(self) -> bool:
        env, i, m, ms = self.env, self.i, self.ms.macro, self.ms
        inv = env.inventory(i)
        if m.kind == "fetch":
            return inv != 0 and decode_ingredients(inv) == [m.arg] and not is_plate(inv)
        if m.kind == "fetch_plate":
            return inv == 1
        if m.kind in ("put_in_pot", "deliver", "place_counter"):
            return inv == 0 and ms.inv0 != 0
        if m.kind == "plate_soup":
            return is_cooked(inv)
        if m.kind == "pick_counter":
            return inv != 0
        if m.kind == "press_button":
            return ms.reason == "pressed"
        return False

    def after_step(self):
        """Call after env.step to mark completion as soon as it happens."""
        if self.ms and self.ms.status == "running" and self.ms.macro.kind not in ("wait", "move_to"):
            if self._complete():
                self.ms.status = "done"
        if self.ms and self.ms.status == "running" and self.ms.macro.kind == "move_to":
            if self.env.agent_pos(self.i) == self.ms.macro.xy:
                self.ms.status = "done"


def pot_side_cell(env: CoopEnv, i: int, pot) -> Optional[tuple]:
    """The floor cell next to `pot` that agent i reaches first (None if unreachable)."""
    adj = {n for n in env.neighbors(pot) if env._walkable(n)}
    path = bfs(env, env.agent_pos(i), adj, set()) if adj else None
    return path[-1] if path else None


def valid_actions(env: CoopEnv, i: int) -> list[str]:
    """Macros that pass the precondition check right now (shown to the LLM; condition-invariant)."""
    ex = Executor(env, i)
    out = []
    cands = [f"fetch {env.ing_name(k)}" for k in sorted(env.piles)] + [
        "fetch plate", "put in pot", "plate soup", "deliver", "place on counter", "press button",
    ]
    for c in env.counters:
        if env.cell(c)[1] != 0 and env.in_view(i, c):
            cands.append(f"pick up from counter {c[0]} {c[1]}")
    for s in cands:
        m = parse_macro(s, env)
        if m and not ex._precheck(m):
            out.append(s)
    out += ["move to <x> <y>", "wait <n>"]
    return out
