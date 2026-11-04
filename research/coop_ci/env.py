"""OvercookedV2 wrapper for LLM agents (cooperative-intelligence experiments).

Wraps this repo's OvercookedV2 (envs/overcooked_v2, from JaxMARL; Gessler et al., ICLR 2025)
and adds what the experiments need on top of the raw env:

* layouts from configs/layouts.yaml, pots named pot_0..k (sorted by x, then y),
* open orders that both cooks see (common knowledge), from a sequence fixed by the seed
  (identical across conditions -> paired design); any open order can be served,
* full view by default (`view_radius=None`); a radius is kept for a later factor,
* step events, per-soup bookkeeping keyed by pot (who added what, when it was ready,
  plated, and which dish was delivered),
* `snapshot()` / `restore()` for branch rollouts from saved states.

Team return is computed here from deliveries only: correct +20, wrong -20 (no shaped
reward). The JAX env is stepped one transition at a time; LLM latency dominates.
"""
from __future__ import annotations

import copy
import os
import threading
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from envs.overcooked_v2 import layouts as L
from envs.overcooked_v2.common import DynamicObject, StaticObject
from envs.overcooked_v2.overcooked import OvercookedV2
from envs.overcooked_v2.settings import POT_COOK_TIME

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")

# Named recipe sets. Recipes are sorted ingredient-index triples (0 onion, 1 tomato).
RECIPE_SETS = {
    "pure_01": [[0, 0, 0], [1, 1, 1]],
    "mixed_two": [[0, 0, 1], [0, 1, 1]],              # every order needs both ingredients (F1)
    "mixed_01": [[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]],
}
DEFAULT_INGREDIENT_NAMES = ["onion", "tomato", "broccoli", "mushroom"]
DELIVERY_REWARD = 20

# OvercookedActionsEnum: right=0, down=1, left=2, up=3, stay=4, interact=5
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5
MOVE_ACTIONS = (A_RIGHT, A_DOWN, A_LEFT, A_UP)
ACTION_VEC = {A_RIGHT: (1, 0), A_DOWN: (0, 1), A_LEFT: (-1, 0), A_UP: (0, -1)}
# Direction enum: UP=0, DOWN=1, RIGHT=2, LEFT=3
DIR_VEC = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0)}
DIR_NAME = {0: "up", 1: "down", 2: "right", 3: "left"}
VEC_TO_ACTION = {(0, -1): A_UP, (0, 1): A_DOWN, (1, 0): A_RIGHT, (-1, 0): A_LEFT}

_JAX_LOCK = threading.Lock()
_LAYOUTS: Optional[dict] = None


def layouts() -> dict:
    global _LAYOUTS
    if _LAYOUTS is None:
        with open(os.path.join(CONFIG_DIR, "layouts.yaml")) as f:
            _LAYOUTS = yaml.safe_load(f)
    return _LAYOUTS


def resolve_recipes(recipes) -> list[list[int]]:
    if isinstance(recipes, str):
        recipes = RECIPE_SETS[recipes]
    return [sorted(int(i) for i in r) for r in recipes]


# --------------------------------------------------------------------------- #
# Encoding helpers
# --------------------------------------------------------------------------- #
def xy(p) -> str:
    return f"({p[0]},{p[1]})"


def decode_ingredients(obj: int) -> list[int]:
    """Ingredient-index multiset from a DynamicObject bitfield (2 bits per ingredient)."""
    obj = int(obj) >> 2
    res, idx = [], 0
    while obj > 0:
        res += [idx] * (obj & 0x3)
        obj >>= 2
        idx += 1
    return res


def encode_recipe(recipe: list[int]) -> int:
    return int(sum(int(DynamicObject.ingredient(i)) for i in recipe))


def is_plate(obj: int) -> bool:
    return bool(int(obj) & int(DynamicObject.PLATE))


def is_cooked(obj: int) -> bool:
    return bool(int(obj) & int(DynamicObject.COOKED))


def is_ingredient(obj: int) -> bool:
    return obj != 0 and not is_plate(obj) and not is_cooked(obj)


def multiset_contains(big: list[int], small: list[int]) -> bool:
    b = list(big)
    for s in small:
        if s in b:
            b.remove(s)
        else:
            return False
    return True


def multiset_minus(big: list[int], small: list[int]) -> list[int]:
    b = list(big)
    for s in small:
        if s in b:
            b.remove(s)
    return b


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class Event:
    t: int
    kind: str  # pot_add | counter_place | counter_pick | fetch | plate_soup | deliver
    agent: int
    pos: tuple[int, int]  # cell acted on
    item: str = ""
    correct: Optional[bool] = None
    extra: dict = field(default_factory=dict)


@dataclass
class Soup:
    idx: int
    pot: str                    # pot name
    start_t: int
    adds: list = field(default_factory=list)  # (t, agent, ingredient)
    contents: list = field(default_factory=list)
    full_t: Optional[int] = None
    ready_t: Optional[int] = None
    plated_t: Optional[int] = None
    plated_by: Optional[int] = None
    delivered_t: Optional[int] = None
    delivered_by: Optional[int] = None
    correct: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Environment wrapper
# --------------------------------------------------------------------------- #
class CoopEnv:
    def __init__(
        self,
        layout: str,
        recipes=None,
        orders_visible: Optional[int] = None,
        view_radius: Optional[int] = None,
        max_steps: int = 300,
        ingredient_names: Optional[list[str]] = None,
    ):
        spec = layouts()[layout]
        self.layout_name = layout
        self.recipes = resolve_recipes(recipes if recipes is not None else spec.get("recipes", "mixed_01"))
        self.n_open = int(orders_visible or spec.get("orders_visible", 1))
        self.layout = L.Layout.from_string(spec["ascii"], possible_recipes=self.recipes)
        self.view_radius = view_radius
        self.max_steps = max_steps
        self.ingredient_names = ingredient_names or DEFAULT_INGREDIENT_NAMES
        # We handle orders and scoring ourselves -> the env never resamples.
        self.env = OvercookedV2(layout=self.layout, max_steps=10**6, agent_view_size=None,
                                negative_rewards=True, sample_recipe_on_delivery=False,
                                random_agent_positions=False)
        self._step_fn = jax.jit(self.env.step_env)
        self.num_agents = self.env.num_agents
        self._analyse_static()

    # ---------------- static layout analysis ---------------- #
    def _analyse_static(self):
        s = np.asarray(self.layout.static_objects)
        self.static = s
        self.H, self.W = s.shape
        self.walkable = s == int(StaticObject.EMPTY)
        self.piles: dict[int, list] = {}
        self.plate_piles, self.pots, self.goals, self.counters = [], [], [], []
        for y in range(self.H):
            for x in range(self.W):
                v = int(s[y, x])
                if StaticObject.is_ingredient_pile(v):
                    self.piles.setdefault(v - int(StaticObject.INGREDIENT_PILE_BASE), []).append((x, y))
                elif v == StaticObject.PLATE_PILE:
                    self.plate_piles.append((x, y))
                elif v == StaticObject.POT:
                    self.pots.append((x, y))
                elif v == StaticObject.GOAL:
                    self.goals.append((x, y))
                elif v == StaticObject.WALL and any(self._walkable(n) for n in self.neighbors((x, y))):
                    self.counters.append((x, y))
        self.pots.sort()
        self.pot_name = {p: f"pot_{k}" for k, p in enumerate(self.pots)}
        self.pot_by_name = {n: p for p, n in self.pot_name.items()}

    def neighbors(self, p):
        x, y = p
        for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.W and 0 <= ny < self.H:
                yield (nx, ny)

    def _walkable(self, p):
        return bool(self.walkable[p[1], p[0]])

    def ing_name(self, i: int) -> str:
        return self.ingredient_names[i]

    def recipe_str(self, recipe: list[int]) -> str:
        return "+".join(self.ing_name(i) for i in sorted(recipe))

    def menu(self) -> list[str]:
        return [self.recipe_str(r) for r in self.recipes]

    def static_map_text(self) -> str:
        sym = {int(StaticObject.EMPTY): ".", int(StaticObject.WALL): "W", int(StaticObject.GOAL): "X",
               int(StaticObject.POT): "P", int(StaticObject.PLATE_PILE): "B"}
        rows = ["     x=" + "".join(str(x % 10) for x in range(self.W))]
        for y in range(self.H):
            line = ""
            for x in range(self.W):
                v = int(self.static[y, x])
                line += str(v - int(StaticObject.INGREDIENT_PILE_BASE)) if StaticObject.is_ingredient_pile(v) else sym.get(v, "?")
            rows.append(f"  y={y:<2} {line}")
        pots = ", ".join(f"{self.pot_name[p]} at {xy(p)}" for p in self.pots)
        legend = (
            "Legend: '.' floor (walkable); W counter/wall (a W next to floor can hold one item); "
            "P pot; X serving window (deliver dishes here); B plate pile; digits = ingredient piles ("
            + ", ".join(f"{i}={self.ing_name(i)}" for i in sorted(self.piles)) + f"). Pots: {pots}."
        )
        return "\n".join(rows) + "\n" + legend

    # ---------------- episode lifecycle ---------------- #
    def reset(self, seed: int, recipe_seq_len: int = 128):
        rng = np.random.default_rng(seed)
        self.recipe_seq = [list(self.recipes[i]) for i in rng.integers(0, len(self.recipes), size=recipe_seq_len)]
        self.orders = [list(r) for r in self.recipe_seq[:self.n_open]]
        self.seq_ptr = self.n_open
        self.key = jax.random.PRNGKey(seed)
        with _JAX_LOCK:
            self.key, k = jax.random.split(self.key)
            _, st = self.env.reset(k)
            st = st.replace(recipe=jnp.array(encode_recipe(self.orders[0]), dtype=st.recipe.dtype))
        self.state = st
        self._sync()
        self.t = 0
        self.score = 0.0
        self.events: list[Event] = []
        self.soups: list[Soup] = []
        self._open_soup: dict = {}   # pot xy -> Soup still in the pot (filling / cooking / ready)
        self._carried: dict = {}     # agent -> Soup on the plate it holds
        self._on_counter: dict = {}  # counter xy -> Soup set down there on a plate
        self.observed: list[deque] = [deque(maxlen=200) for _ in range(self.num_agents)]
        self.order_history = [(0, [list(o) for o in self.orders])]
        return self

    _PY_FIELDS = ("t", "score", "recipe_seq", "orders", "seq_ptr", "events", "soups", "_open_soup",
                  "_carried", "_on_counter", "observed", "order_history")

    def snapshot(self) -> dict:
        """Everything needed to continue this episode, except the jitted step function and the
        JaxMARL env object. JAX arrays are immutable, so they are shared, not copied."""
        return {"state": self.state, "key": self.key,
                "py": copy.deepcopy({k: getattr(self, k) for k in self._PY_FIELDS})}

    def restore(self, snap: dict):
        self.state, self.key = snap["state"], snap["key"]
        for k, v in copy.deepcopy(snap["py"]).items():
            setattr(self, k, v)
        self._sync()
        return self

    @property
    def done(self) -> bool:
        return self.t >= self.max_steps

    # ---------------- state accessors (numpy mirrors; indexing jax arrays per call is slow) ---------------- #
    def _sync(self):
        a = self.state.agents
        self._px, self._py = np.asarray(a.pos.x).astype(int), np.asarray(a.pos.y).astype(int)
        self._dir, self._inv = np.asarray(a.dir).astype(int), np.asarray(a.inventory).astype(int)
        self._grid = np.asarray(self.state.grid).astype(int)

    def agent_pos(self, i):
        return int(self._px[i]), int(self._py[i])

    def agent_dir(self, i):
        return int(self._dir[i])

    def inventory(self, i) -> int:
        return int(self._inv[i])

    def cell(self, p):
        c = self._grid[p[1], p[0]]
        return int(c[0]), int(c[1]), int(c[2])

    def pot_status(self, p):
        """('idle' | 'cooking' | 'ready', contents, steps left)."""
        _, ing, extra = self.cell(p)
        contents = decode_ingredients(ing)
        if is_cooked(ing):
            status = "ready"
        elif extra > 0:
            status = "cooking"
        else:
            status = "idle"
        return status, contents, extra

    def in_view(self, i, p) -> bool:
        if self.view_radius is None:
            return True
        ax, ay = self.agent_pos(i)
        return max(abs(p[0] - ax), abs(p[1] - ay)) <= self.view_radius

    def item_text(self, obj: int) -> str:
        if obj == 0:
            return "nothing"
        ings = decode_ingredients(obj)
        if is_plate(obj) and is_cooked(obj):
            return f"a dish of {self.recipe_str(ings)} soup"
        if is_plate(obj):
            return "an empty plate"
        if len(ings) == 1:
            n = self.ing_name(ings[0])
            return f"{'an' if n[0] in 'aeiou' else 'a'} {n}"
        return "+".join(self.ing_name(i) for i in ings)

    def counter_items(self) -> dict:
        return {c: self.cell(c)[1] for c in self.counters if self.cell(c)[1]}

    def matches_open_order(self, contents: list[int]) -> bool:
        return any(sorted(contents) == sorted(o) for o in self.orders)

    # ---------------- which pot serves which open order ---------------- #
    def pot_plan(self) -> list[dict]:
        """Assign open orders to pots (common rule for every scripted policy and metric).

        Dishes already plated (carried or on a counter) claim their order first, then pots
        that hold ingredients (fullest first) claim the first order their contents fit, then
        empty pots take the remaining orders. A pot whose contents fit no open order gets
        target None (a wrong soup)."""
        free = list(range(len(self.orders)))

        def claim(contents):
            for k in free:
                if multiset_contains(self.orders[k], contents):
                    free.remove(k)
                    return list(self.orders[k])
            return None

        dishes = [self.inventory(i) for i in range(self.num_agents) if is_cooked(self.inventory(i))]
        dishes += [o for o in self.counter_items().values() if is_cooked(o)]
        for d in dishes:
            claim(decode_ingredients(d))
        info = {p: self.pot_status(p) for p in self.pots}
        out = {}
        for p in sorted(self.pots, key=lambda q: (-len(info[q][1]), q)):
            status, contents, extra = info[p]
            if contents:
                target = claim(contents)
            else:
                target = list(self.orders[free.pop(0)]) if free else None
            missing = multiset_minus(target, contents) if (target is not None and status == "idle") else []
            out[p] = {"pot": p, "name": self.pot_name[p], "status": status, "contents": contents,
                      "steps_left": extra, "target": target, "missing": missing}
        return [out[p] for p in self.pots]

    def missing_ingredients(self, subtract_held_by=()) -> Counter:
        """Ingredients the open pots still need, minus those held by the given agents."""
        need = Counter()
        for pp in self.pot_plan():
            need.update(pp["missing"])
        for j in subtract_held_by:
            inv = self.inventory(j)
            if is_ingredient(inv):
                for k in decode_ingredients(inv):
                    if need[k] > 0:
                        need[k] -= 1
        return +need

    def plates_out(self) -> int:
        """Empty plates in hands or on counters."""
        n = sum(1 for i in range(self.num_agents) if self.inventory(i) == int(DynamicObject.PLATE))
        return n + sum(1 for o in self.counter_items().values() if o == int(DynamicObject.PLATE))

    # ---------------- stepping ---------------- #
    def step(self, actions: dict[int, int]):
        prev = self.state
        prev_inv = [self.inventory(i) for i in range(self.num_agents)]
        prev_cells = {}
        for i in range(self.num_agents):
            fx, fy = self._fwd(i)
            if 0 <= fx < self.W and 0 <= fy < self.H:
                prev_cells[i] = self.cell((fx, fy))
        acts = {f"agent_{i}": jnp.array(actions.get(i, A_STAY)) for i in range(self.num_agents)}
        with _JAX_LOCK:
            self.key, k = jax.random.split(self.key)
            _, st, _, _, _ = self._step_fn(k, prev, acts)
        self.state = st
        self._sync()
        self.t += 1
        score0 = self.score
        new_events = self._detect_events(actions, prev_inv, prev_cells)
        self.events.extend(new_events)
        for e in new_events:
            for j in range(self.num_agents):
                if j == e.agent or self.in_view(j, e.pos) or self.in_view(j, self.agent_pos(e.agent)):
                    self.observed[j].append(e)
        return self.score - score0, new_events

    def _fwd(self, i):
        x, y = self.agent_pos(i)
        dx, dy = DIR_VEC[self.agent_dir(i)]
        return x + dx, y + dy

    def _detect_events(self, actions, prev_inv, prev_cells) -> list[Event]:
        evs = []
        for i in range(self.num_agents):
            if actions.get(i) != A_INTERACT or i not in prev_cells:
                continue
            p = self._fwd(i)
            static, ing, extra = self.cell(p)
            _, p_ing, p_extra = prev_cells[i]
            inv0, inv1 = prev_inv[i], self.inventory(i)
            if inv0 == inv1 and ing == p_ing and extra == p_extra:
                continue
            if static == StaticObject.POT and is_ingredient(inv0) and inv1 == 0:
                added = decode_ingredients(inv0)[0]
                name = self.pot_name[p]
                evs.append(Event(self.t, "pot_add", i, p, self.ing_name(added), extra={"ingredient": added, "pot": name}))
                self._on_pot_add(p, i, added)
            elif static == StaticObject.POT and is_plate(inv0) and is_cooked(inv1):
                evs.append(Event(self.t, "plate_soup", i, p, self.item_text(inv1), extra={"pot": self.pot_name[p]}))
                s = self._open_soup.pop(p, None)
                if s:
                    s.plated_t, s.plated_by = self.t, i
                    self._carried[i] = s
            elif static == StaticObject.GOAL and is_cooked(inv0) and inv1 == 0:
                delivered = sorted(decode_ingredients(inv0))
                ok = self._serve(delivered)
                evs.append(Event(self.t, "deliver", i, p, self.item_text(inv0), correct=ok))
                s = self._carried.pop(i, None)
                if s:
                    s.delivered_t, s.delivered_by, s.correct = self.t, i, ok
            elif static == StaticObject.WALL and inv0 != 0 and inv1 == 0:
                evs.append(Event(self.t, "counter_place", i, p, self.item_text(inv0)))
                if i in self._carried:
                    self._on_counter[p] = self._carried.pop(i)
            elif static == StaticObject.WALL and inv0 == 0 and inv1 != 0:
                evs.append(Event(self.t, "counter_pick", i, p, self.item_text(inv1)))
                if p in self._on_counter:
                    self._carried[i] = self._on_counter.pop(p)
            elif StaticObject.is_ingredient_pile(static) or static == StaticObject.PLATE_PILE:
                if inv0 == 0 and inv1 != 0:
                    evs.append(Event(self.t, "fetch", i, p, self.item_text(inv1)))
        for p, s in self._open_soup.items():
            status, contents, _ = self.pot_status(p)
            if s.full_t is None and len(contents) == 3:
                s.full_t = self.t
            if s.ready_t is None and status == "ready":
                s.ready_t = self.t
        return evs

    def _serve(self, dish: list[int]) -> bool:
        """Score a delivery: an open order it matches is closed and the next one opens."""
        for k, o in enumerate(self.orders):
            if sorted(o) == dish:
                self.score += DELIVERY_REWARD
                self.orders.pop(k)
                self.orders.append(list(self.recipe_seq[self.seq_ptr % len(self.recipe_seq)]))
                self.seq_ptr += 1
                self.state = self.state.replace(
                    recipe=jnp.array(encode_recipe(self.orders[0]), dtype=self.state.recipe.dtype))
                self.order_history.append((self.t, [list(x) for x in self.orders]))
                return True
        self.score -= DELIVERY_REWARD
        return False

    def _on_pot_add(self, p, agent, ingredient):
        s = self._open_soup.get(p)
        if s is None:
            s = Soup(idx=len(self.soups), pot=self.pot_name[p], start_t=self.t)
            self.soups.append(s)
            self._open_soup[p] = s
        s.adds.append((self.t, agent, ingredient))
        s.contents = self.pot_status(p)[1]

    # ---------------- text observation ---------------- #
    def orders_text(self) -> str:
        if len(self.orders) == 1:
            return f"Current order: {self.recipe_str(self.orders[0])} soup."
        return ("Open orders (serve any of them): "
                + "; ".join(f"{k + 1}) {self.recipe_str(o)}" for k, o in enumerate(self.orders)) + ".")

    def observation_text(self, i: int, n_events: int = 10) -> str:
        """What cook i sees. Always 'you' / 'partner', never agent indices."""
        x, y = self.agent_pos(i)
        lines = [
            f"Time step {self.t} of {self.max_steps}. Team score so far: {self.score:+.0f}.",
            self.orders_text(),
            f"You: at ({x},{y}) facing {DIR_NAME[self.agent_dir(i)]}, holding {self.item_text(self.inventory(i))}.",
        ]
        for j in range(self.num_agents):
            if j == i:
                continue
            if self.in_view(i, self.agent_pos(j)):
                px, py = self.agent_pos(j)
                lines.append(f"Partner: at ({px},{py}) facing {DIR_NAME[self.agent_dir(j)]}, holding {self.item_text(self.inventory(j))}.")
            else:
                lines.append("Partner: not in view.")
        for p in self.pots:
            if self.in_view(i, p):
                status, contents, extra = self.pot_status(p)
                lines.append(f"{self.pot_name[p]} {xy(p)}: {self.pot_text(status, contents, extra)}{self.fit_text(contents)}.")
            else:
                lines.append(f"{self.pot_name[p]} {xy(p)}: not in view.")
        items = [f"{xy(c)}: {self.item_text(o)}" for c, o in self.counter_items().items() if self.in_view(i, c)]
        lines.append("Items on counters: " + ("; ".join(items) if items else "none") + ".")
        ev = list(self.observed[i])[-n_events:]
        if ev:
            lines.append("Recent events (most recent last):")
            lines += ["  " + self.event_text(e, i) for e in ev]
        return "\n".join(lines)

    def fit_text(self, contents: list[int]) -> str:
        """Which open orders a pot's contents still fit (derived from common knowledge; the same
        for both cooks and every condition)."""
        if not contents:
            return ""
        fits = [k + 1 for k, o in enumerate(self.orders) if multiset_contains(o, contents)]
        if not fits:
            return " - fits no open order (wrong soup)"
        if len(self.orders) == 1:
            return " - fits the order"
        return f" - fits order {' or '.join(map(str, fits))}"

    def pot_text(self, status, contents, extra):
        names = ", ".join(self.ing_name(c) for c in contents) or "empty"
        if status == "cooking":
            return f"cooking [{names}], {extra} steps left"
        if status == "ready":
            return f"soup ready [{names}]"
        return f"{len(contents)}/3 ingredients [{names}]"

    def event_text(self, e: Event, viewer: int) -> str:
        who = "you" if e.agent == viewer else "partner"
        pot = e.extra.get("pot", "the pot")
        d = {
            "pot_add": f"{who} put {e.item} into {pot}",
            "counter_place": f"{who} placed {e.item} on counter {xy(e.pos)}",
            "counter_pick": f"{who} picked up {e.item} from counter {xy(e.pos)}",
            "fetch": f"{who} took {e.item} from {xy(e.pos)}",
            "plate_soup": f"{who} put the soup from {pot} on a plate",
            "deliver": f"{who} delivered {e.item}: {'CORRECT (+20)' if e.correct else 'WRONG (-20)'}",
        }
        return f"[t={e.t}] " + d.get(e.kind, e.kind)


def station_distances(env: CoopEnv, start) -> dict:
    """BFS steps from `start` to the nearest floor cell next to each station type (equal-access test)."""
    dist = {start: 0}
    q = deque([start])
    while q:
        c = q.popleft()
        for n in env.neighbors(c):
            if n not in dist and env._walkable(n):
                dist[n] = dist[c] + 1
                q.append(n)

    def nearest(cells):
        d = [dist[n] for c in cells for n in env.neighbors(c) if n in dist]
        return min(d) if d else None

    out = {f"pile_{env.ing_name(k)}": nearest(v) for k, v in sorted(env.piles.items())}
    out.update(pot=nearest(env.pots), plate=nearest(env.plate_piles), serve=nearest(env.goals))
    return out


__all__ = ["CoopEnv", "Event", "Soup", "POT_COOK_TIME", "station_distances", "multiset_contains", "multiset_minus"]
