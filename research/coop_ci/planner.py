"""Macro-actions on top of OvercookedV2 primitive actions.

Cooks choose macro-actions ("fetch onion", "put in pot", ...); a BFS executor turns them
into primitive moves. A cook is asked again only when its macro finishes or fails.

Every committed macro gets a canonical subgoal label (coarse, fine), e.g. ("load", "load:pot_1"),
with the pot the executor resolved. Failures carry a code from a fixed taxonomy so infrastructure
failures are never mistaken for partner-modelling failures:
  precheck:<what>   the macro could not start (e.g. precheck:hands_full)
  timeout           it did not finish within its step budget
  target_gone       its target disappeared while walking (e.g. the pot filled up)
  deadlock          no progress for DEADLOCK_STEPS steps while blocked by the other cook
  pot_changed       (a replan, not a failure) the pot changed while walking to add an ingredient
  taken_by_partner  (contention, not infrastructure) the partner acted on the target first
"""
from __future__ import annotations

import copy
import random
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .env import A_INTERACT, A_STAY, DIR_VEC, VEC_TO_ACTION, CoopEnv, decode_ingredients, is_cooked, is_ingredient, is_plate

MAX_WAIT = 10
DEFAULT_TIMEOUT = 40
DEADLOCK_STEPS = 12
NOT_INFRA = ("pot_changed", "taken_by_partner")


def grammar_for(env: CoopEnv) -> str:
    many = len(env.pots) > 1
    pot = " [pot_k]" if many else ""
    lines = [
        "- fetch <ingredient>          take one ingredient from its pile (hands must be empty)",
        "- fetch plate                 take an empty plate from a plate pile B (hands must be empty)",
        f"- put in pot{pot:<19}put the held ingredient into " + ("pot_k (or the nearest pot that can take it)" if many else "the pot"),
        f"- plate soup{pot:<19}with an empty plate in hand, take the soup from a pot that is cooking (waits by it) or ready",
        "- deliver                     bring the held dish to the serving window X",
        "- put back                    return the held ingredient or empty plate to its pile",
        "- place on counter <x> <y>    put the held item on that counter (omit x y for the nearest free one)",
        "- pick up from counter <x> <y>",
        "- move to <x> <y>             walk to a floor cell, or next to the station/counter at (x, y)",
        "- wait <n>                    stay put for n steps (1-10)",
    ]
    return "\n".join(lines)


@dataclass
class Macro:
    kind: str
    arg: Optional[int] = None          # ingredient index (fetch)
    pot: Optional[str] = None          # pot name (put_in_pot / plate_soup)
    xy: Optional[tuple[int, int]] = None
    n: int = 1
    raw: str = ""

    def __str__(self):
        return self.raw or self.kind


def parse_macro(text: str, env: CoopEnv) -> Optional[Macro]:
    s = text.strip().lower().strip(".").replace("_", " ")
    s = re.sub(r"[(),]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    m = re.fullmatch(r"fetch (?:an? |one )?(\w+)", s)
    if m:
        name = m.group(1)
        if name in ("plate", "dish", "bowl"):
            return Macro("fetch_plate", raw="fetch plate")
        names = [n.lower() for n in env.ingredient_names]
        if name in names and names.index(name) in env.piles:
            return Macro("fetch", arg=names.index(name), raw=f"fetch {name}")
        return None

    def pot_arg(g):
        if g is None:
            return None, ""
        name = f"pot_{int(g)}"
        return (name, name) if name in env.pot_by_name else (False, "")

    m = re.fullmatch(r"put (?:it )?in(?:to)? (?:the )?pot(?: pot)?(?: (\d+))?", s)
    if m:
        pot, name = pot_arg(m.group(1))
        if pot is False:
            return None
        return Macro("put_in_pot", pot=pot, raw=f"put in {name}" if pot else "put in pot")
    m = re.fullmatch(r"plate (?:the )?soup(?: (?:from |in |at )?(?:the )?pot(?: pot)?(?: (\d+))?)?", s)
    if m:
        pot, name = pot_arg(m.group(1))
        if pot is False:
            return None
        return Macro("plate_soup", pot=pot, raw=f"plate soup {name}" if pot else "plate soup")
    if re.fullmatch(r"(?:put (?:it )?back|return(?: it)?)(?: (?:to|on) (?:the |its )?pile)?", s):
        return Macro("put_back", raw="put back")
    if re.fullmatch(r"deliver(?: (?:the )?dish)?", s):
        return Macro("deliver", raw="deliver")
    m = re.fullmatch(r"place (?:it )?on (?:the )?counter(?: (\d+) (\d+))?", s)
    if m:
        p = (int(m.group(1)), int(m.group(2))) if m.group(1) else None
        return Macro("place_counter", xy=p, raw=f"place on counter {p[0]} {p[1]}" if p else "place on counter")
    m = re.fullmatch(r"pick up (?:from )?(?:the )?counter (\d+) (\d+)", s)
    if m:
        p = (int(m.group(1)), int(m.group(2)))
        return Macro("pick_counter", xy=p, raw=f"pick up from counter {p[0]} {p[1]}")
    m = re.fullmatch(r"move to (\d+) (\d+)", s)
    if m:
        p = (int(m.group(1)), int(m.group(2)))
        return Macro("move_to", xy=p, raw=f"move to {p[0]} {p[1]}")
    m = re.fullmatch(r"wait(?: (\d+))?(?: steps?)?", s)
    if m:
        n = max(1, min(MAX_WAIT, int(m.group(1)) if m.group(1) else 1))
        return Macro("wait", n=n, raw=f"wait {n}")
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
    """Targets (non-walkable cells) that cook i can stand next to; sorted by path length."""
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


def side_cell(env: CoopEnv, i: int, target) -> Optional[tuple]:
    """The floor cell next to `target` that cook i reaches first (None if unreachable)."""
    adj = {n for n in env.neighbors(target) if env._walkable(n)}
    path = bfs(env, env.agent_pos(i), adj, set()) if adj else None
    return path[-1] if path else None


def item_kind(env: CoopEnv, obj: int) -> str:
    """'onion' / 'tomato' / 'plate' / 'dish' / 'nothing' (for subgoal labels and compliance)."""
    if obj == 0:
        return "nothing"
    if is_cooked(obj):
        return "dish"
    if is_plate(obj):
        return "plate"
    ings = decode_ingredients(obj)
    return env.ing_name(ings[0]) if len(ings) == 1 else "mixed"


def subgoal(env: CoopEnv, i: int, m: Macro, target=None) -> tuple[str, str]:
    """Canonical (coarse, fine) label of a committed macro; fine labels name the resolved pot."""
    if m.kind == "fetch":
        lab = f"fetch:{env.ing_name(m.arg)}"
        return lab, lab
    if m.kind == "fetch_plate":
        return "fetch:plate", "fetch:plate"
    if m.kind in ("put_in_pot", "plate_soup"):
        coarse = "load" if m.kind == "put_in_pot" else "plate"
        pot = env.pot_name.get(target) if target is not None else m.pot
        return coarse, f"{coarse}:{pot or '?'}"
    if m.kind == "deliver":
        return "deliver", "deliver"
    if m.kind == "place_counter":
        return "counter:place", f"counter:place:{item_kind(env, env.inventory(i))}"
    if m.kind == "pick_counter":
        obj = env.cell(m.xy)[1] if m.xy and m.xy in env.counters else 0
        return "counter:pick", f"counter:pick:{item_kind(env, obj)}"
    if m.kind == "move_to":
        return "move", "move"
    if m.kind == "put_back":
        return "put_back", f"put_back:{item_kind(env, env.inventory(i))}"
    return "wait", "wait"


def acquires(env: CoopEnv, m: Macro) -> Optional[str]:
    """What a macro picks up, if anything: an ingredient name, 'plate' or 'dish'."""
    if m.kind == "fetch":
        return env.ing_name(m.arg)
    if m.kind == "fetch_plate":
        return "plate"
    if m.kind == "pick_counter" and m.xy in env.counters:
        return item_kind(env, env.cell(m.xy)[1])
    return None


@dataclass
class MacroState:
    macro: Macro
    started_t: int
    timeout: int = DEFAULT_TIMEOUT
    steps: int = 0
    status: str = "running"  # running | done | failed | failed_reported
    code: str = ""           # failure code (taxonomy above)
    reason: str = ""         # human-readable reason (shown to the cook)
    note: str = ""           # shown with the result when the macro finished differently than named
    target: Optional[tuple] = None
    subgoal: tuple = ("wait", "wait")
    inv0: int = 0
    pots_at_start: dict = field(default_factory=dict)
    blocked: int = 0
    stuck: int = 0


class Executor:
    """Turns the current macro of cook i into one primitive action per step."""

    def __init__(self, env: CoopEnv, i: int, salt: int = 0):
        self.env, self.i, self.salt = env, i, salt
        self.ms: Optional[MacroState] = None
        self.intended: Optional[tuple] = None  # next cell this cook wants to enter (for yielding)
        self._moved_from: Optional[tuple] = None
        self._hist: deque = deque(maxlen=4)    # own recent positions (livelock detection)

    def get_state(self) -> dict:
        return copy.deepcopy({"ms": self.ms, "intended": self.intended, "_moved_from": self._moved_from,
                              "salt": self.salt, "_hist": self._hist})

    def set_state(self, st: dict):
        st = copy.deepcopy(st)
        self.ms, self.intended, self._moved_from, self.salt = st["ms"], st["intended"], st["_moved_from"], st["salt"]
        self._hist = st["_hist"]

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
        env = self.env
        self.ms = MacroState(macro, env.t, inv0=env.inventory(self.i),
                             pots_at_start={p: env.pot_status(p)[1] for p in env.pots})
        code, err = self._precheck(macro)
        if err:
            self.ms.status, self.ms.code, self.ms.reason = "failed", f"precheck:{code}", err
        tg = self._targets(macro) if macro.kind not in ("wait", "move_to") else []
        self.ms.target = tg[0] if tg else None
        if macro.kind == "place_counter" and macro.xy is not None and tg and tg[0] != macro.xy:
            self.ms.note = f"{macro.xy} is not a free counter, so you used the nearest free one at {tg[0]}"
        self.ms.subgoal = subgoal(env, self.i, macro, self.ms.target)

    def idle(self) -> bool:
        return self.ms is None or self.ms.status != "running"

    # ---- validity ---- #
    def _precheck(self, m: Macro) -> tuple[str, str]:
        env, i = self.env, self.i
        inv = env.inventory(i)
        if m.kind in ("fetch", "fetch_plate", "pick_counter") and inv != 0:
            return "hands_full", f"hands are not empty (holding {env.item_text(inv)})"
        if m.kind == "put_in_pot" and not is_ingredient(inv):
            return "no_ingredient", "not holding an ingredient"
        if m.kind == "plate_soup" and not (is_plate(inv) and not is_cooked(inv)):
            return "no_plate", "not holding an empty plate"
        if m.kind == "deliver" and not is_cooked(inv):
            return "no_dish", "not holding a dish"
        if m.kind == "place_counter" and inv == 0:
            return "empty_hands", "holding nothing"
        if m.kind == "put_back" and not (is_ingredient(inv) or (is_plate(inv) and not is_cooked(inv))):
            return ("no_item", "holding nothing" if inv == 0 else
                    f"only an ingredient or an empty plate can be put back (holding {env.item_text(inv)})")
        if m.kind == "move_to":
            if m.xy is None or not (0 <= m.xy[0] < env.W and 0 <= m.xy[1] < env.H):
                return "outside", f"{m.xy} is outside the kitchen"
            goals = self._move_goals(m.xy)
            if not goals or bfs(env, env.agent_pos(i), goals, set()) is None:
                return "unreachable", f"you cannot reach {m.xy} or any floor cell next to it"
            return "", ""
        if m.kind != "wait" and not self._targets(m):
            return "no_target", self._no_target_reason(m)
        return "", ""

    def _move_goals(self, p) -> set:
        env = self.env
        if env._walkable(p):
            return {p}
        return {n for n in env.neighbors(p) if env._walkable(n)}

    def _no_target_reason(self, m: Macro) -> str:
        env = self.env
        pots = "; ".join(f"{env.pot_name[p]}: {self._pot_brief(p)}" for p in env.pots)
        if m.kind == "plate_soup":
            which = m.pot or "a pot"
            return f"plate soup needs {which} cooking or ready ({pots})"
        if m.kind == "put_in_pot":
            which = f"{m.pot} cannot" if m.pot else "no pot can"
            return f"{which} take an ingredient now ({pots})"
        if m.kind == "place_counter":
            free = [c for c in env.counters if env.cell(c)[1] == 0]
            reach = reachable_targets(env, self.i, free)
            options = (f"free counters you can reach: {', '.join(f'{c[0]} {c[1]}' for c in reach[:6])}"
                       if reach else "every counter you can reach already holds an item")
            if m.xy is not None and m.xy not in env.counters:
                return f"{m.xy} is not a counter ({options})"
            if m.xy is not None:
                return f"counter {m.xy} is occupied or out of reach ({options})"
            return f"no free counter you can reach ({options})"
        if m.kind == "pick_counter":
            return f"there is nothing you can pick up on counter {m.xy}"
        if m.kind in ("fetch", "fetch_plate"):
            return "you cannot reach that pile from here"
        return "no reachable target for this action"

    def _pot_brief(self, p) -> str:
        status, contents, extra = self.env.pot_status(p)
        names = ", ".join(self.env.ing_name(c) for c in contents) or "empty"
        if status == "idle":
            return f"filling {len(contents)}/3 [{names}]"
        return f"cooking [{names}], {extra} steps left" if status == "cooking" else f"ready [{names}]"

    def _targets(self, m: Macro) -> list:
        env, i = self.env, self.i
        pots = [env.pot_by_name[m.pot]] if m.pot else env.pots
        if m.kind == "fetch":
            cand = env.piles.get(m.arg, [])
        elif m.kind == "fetch_plate":
            cand = env.plate_piles
        elif m.kind == "put_in_pot":
            cand = [p for p in pots if env.pot_status(p)[0] == "idle" and len(env.pot_status(p)[1]) < 3]
        elif m.kind == "plate_soup":
            st = {p: env.pot_status(p) for p in pots}
            ready = [p for p in pots if st[p][0] == "ready"]
            cand = ready or [p for p in pots if st[p][0] == "cooking"]
        elif m.kind == "deliver":
            cand = env.goals
        elif m.kind == "put_back":
            inv = env.inventory(i)
            cand = env.plate_piles if is_plate(inv) else env.piles.get(decode_ingredients(inv)[0], []) if inv else []
        elif m.kind == "place_counter":
            # A named cell that is not a free counter falls back to the nearest free one: the
            # intent ("put this down") is clear, and the result message says where it went.
            free = [c for c in env.counters if env.cell(c)[1] == 0]
            cand = [m.xy] if m.xy in free else free
        elif m.kind == "pick_counter":
            cand = [m.xy] if (m.xy in env.counters and env.cell(m.xy)[1] != 0) else []
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
            ms.status, ms.code, ms.reason = "failed", "timeout", "timed out"
            return A_STAY
        if ms.stuck >= DEADLOCK_STEPS:
            ms.status, ms.code, ms.reason = "failed", "deadlock", "blocked by your partner for too long"
            return A_STAY
        me = env.agent_pos(i)
        other = {env.agent_pos(j) for j in range(env.num_agents) if j != i}
        if m.kind == "move_to":
            goals = self._move_goals(m.xy)
            if me in goals:
                ms.status = "done"
                return A_STAY
            path = bfs(env, me, goals, other) or bfs(env, me, goals, set())
            return self._step_along(path, tried_from)
        if self._complete():
            ms.status = "done"
            return A_STAY
        targets = self._targets(m)
        if not targets:
            # Contention (the partner took the soup / filled the pot / emptied the counter we were
            # heading to) is coordination, not infrastructure: it gets its own code.
            taken = ms.target is not None and any(
                e.agent != i and e.pos == ms.target and e.t >= ms.started_t for e in env.events[-8:])
            ms.status, ms.code = "failed", "taken_by_partner" if taken else "target_gone"
            ms.reason = ("your partner got there first (" if taken else "target no longer available (") \
                + self._no_target_reason(m) + ")"
            return A_STAY
        t = targets[0]
        ms.target = t
        adj = {n for n in env.neighbors(t) if env._walkable(n)}
        if me in adj:
            ms.stuck = 0
            need = (t[0] - me[0], t[1] - me[1])
            if DIR_VEC[env.agent_dir(i)] != need:
                return VEC_TO_ACTION[need]  # blocked move = turn to face
            if m.kind == "put_in_pot" and env.pot_status(t)[1] != ms.pots_at_start.get(t):
                # The pot changed while walking (e.g. the partner added an ingredient): hand control
                # back so the cook decides with what it now sees. Same rule for every cook.
                now = ", ".join(env.ing_name(c) for c in env.pot_status(t)[1]) or "empty"
                ms.status, ms.code = "failed", "pot_changed"
                ms.reason = f"{env.pot_name[t]} changed while you were walking (now: {now})"
                return A_STAY
            if m.kind == "plate_soup" and env.pot_status(t)[0] == "cooking":
                return A_STAY  # wait by the pot
            return A_INTERACT
        path = bfs(env, me, adj, other) or bfs(env, me, adj, set())
        return self._step_along(path, tried_from)

    def _step_along(self, path, tried_from=None) -> int:
        if not path or len(path) < 2:
            self.ms.stuck += 1
            return A_STAY
        a, b = path[0], path[1]
        others = {self.env.agent_pos(j) for j in range(self.env.num_agents) if j != self.i}
        h = self._hist
        h.append(a)
        if len(h) == 4 and h[0] == h[2] and h[1] == h[3] and h[0] != h[1]:
            # Oscillating (A-B-A-B): both cooks keep detouring the same way in mirror image.
            # Pause at random (seeded per cook) to break the symmetry; counts towards deadlock.
            self.ms.stuck += 1
            if self._rng().random() < 0.5:
                h.clear()
                return A_STAY
        # Blocked = partner stands on the next cell, or last step's move from here did not happen
        # (both stepped into the same cell). Break the symmetry with a random sidestep.
        if b in others or tried_from == a:
            self.ms.blocked += 1
            self.ms.stuck += 1
            if self.ms.blocked >= 2 and self._rng().random() < 0.5:
                self.ms.blocked = 0
                side = self.sidestep(avoid=set())
                if side is not None:
                    return side
        else:
            self.ms.blocked = 0
            self.ms.stuck = 0
        self.intended = b
        self._moved_from = a
        return VEC_TO_ACTION[(b[0] - a[0], b[1] - a[1])]

    def _complete(self) -> bool:
        env, i, m, ms = self.env, self.i, self.ms.macro, self.ms
        inv = env.inventory(i)
        if m.kind == "fetch":
            return is_ingredient(inv) and decode_ingredients(inv) == [m.arg]
        if m.kind == "fetch_plate":
            return is_plate(inv) and not is_cooked(inv)
        if m.kind in ("put_in_pot", "deliver", "place_counter", "put_back"):
            return inv == 0 and ms.inv0 != 0
        if m.kind == "plate_soup":
            return is_cooked(inv)
        if m.kind == "pick_counter":
            return inv != 0
        return False

    def after_step(self):
        """Call after env.step to mark completion as soon as it happens."""
        ms = self.ms
        if ms and ms.status == "running" and ms.macro.kind not in ("wait", "move_to") and self._complete():
            ms.status = "done"
        if ms and ms.status == "running" and ms.macro.kind == "move_to":
            if self.env.agent_pos(self.i) in self._move_goals(ms.macro.xy):
                ms.status = "done"


def valid_actions(env: CoopEnv, i: int) -> list[str]:
    """Macros that pass the precondition check right now (shown to LLM cooks; condition-invariant)."""
    ex = Executor(env, i)
    many = len(env.pots) > 1
    cands = [f"fetch {env.ing_name(k)}" for k in env.in_display_order(env.piles)] + ["fetch plate"]
    cands += [f"put in {env.pot_name[p]}" for p in env.pots] if many else ["put in pot"]
    cands += [f"plate soup {env.pot_name[p]}" for p in env.pots] if many else ["plate soup"]
    cands += ["deliver", "place on counter", "put back"]
    for c in env.counters:
        if env.cell(c)[1] != 0 and env.in_view(i, c):
            cands.append(f"pick up from counter {c[0]} {c[1]}")
    out = []
    for s in cands:
        m = parse_macro(s, env)
        if m and not ex._precheck(m)[1]:
            out.append(s)
    return out + ["move to <x> <y>", "wait <n>"]
