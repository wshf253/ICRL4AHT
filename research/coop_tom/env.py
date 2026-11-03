"""OvercookedV2 wrapper for LLM agents (cooperative-intelligence experiments).

Wraps this repo's OvercookedV2 (envs/overcooked_v2, from JaxMARL; Gessler et al.,
ICLR 2025) and adds what the LLM experiments need on top of the raw env:

* our own recipe sequence per episode (identical across conditions -> paired design),
* partial-observability-respecting views (Chebyshev radius, same as `agent_view_size`),
* step events (pot adds, counter place/pick, deliveries, button presses),
* per-soup bookkeeping (consistent recipe set over time, who added what, phases).

The JAX env is stepped one transition at a time; nothing here is vectorised,
because LLM latency dominates.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from envs.overcooked_v2 import layouts as L
from envs.overcooked_v2.common import DynamicObject, StaticObject
from envs.overcooked_v2.overcooked import OvercookedV2

# --------------------------------------------------------------------------- #
# Layouts and recipe sets
# --------------------------------------------------------------------------- #
LAYOUT_STRINGS = {
    # Overcooked-AI cramped room with an order display R next to agent 1's start
    # (stands in for JaxMARL 0.2's cramped_room_v2, which this repo does not have).
    "cramped_room_up": L.cramped_room_up,
    "grounded_coord_simple": L.grounded_coord_simple,
    "grounded_coord_ring": L.grounded_coord_ring,
    "test_time_simple": L.test_time_simple,
    "test_time_wide": L.test_time_wide,
    "demo_cook_simple": L.demo_cook_simple,
    "demo_cook_wide": L.demo_cook_wide,
}

# Named recipe sets. Recipes are sorted ingredient-index triples.
RECIPE_SETS = {
    # JaxMARL defaults for the coordination layouts: the first ingredient gives the answer away.
    "pure_01": [[0, 0, 0], [1, 1, 1]],
    # All 3-multisets of {0, 1}: one placement leaves 2-3 candidates -> evidence accumulates.
    "mixed_01": [[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]],
    # Held-out set for Demo Cook: every order needs ingredient 2, which only the
    # informed partner can reach -> a different division of labour at test time.
    "heldout_2": [[0, 0, 2], [0, 1, 2], [1, 1, 2], [0, 2, 2]],
}

DEFAULT_INGREDIENT_NAMES = ["onion", "tomato", "broccoli", "mushroom"]

# OvercookedActionsEnum: right=0, down=1, left=2, up=3, stay=4, interact=5
A_RIGHT, A_DOWN, A_LEFT, A_UP, A_STAY, A_INTERACT = 0, 1, 2, 3, 4, 5
# Direction enum: UP=0, DOWN=1, RIGHT=2, LEFT=3
DIR_VEC = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0)}
DIR_NAME = {0: "up", 1: "down", 2: "right", 3: "left"}
VEC_TO_ACTION = {(0, -1): A_UP, (0, 1): A_DOWN, (1, 0): A_RIGHT, (-1, 0): A_LEFT}

_JAX_LOCK = threading.Lock()


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
    kind: str  # pot_add | counter_place | counter_pick | fetch | plate_soup | deliver | button
    agent: int
    pos: tuple[int, int]  # cell acted on
    item: str = ""
    correct: Optional[bool] = None
    extra: dict = field(default_factory=dict)


@dataclass
class Soup:
    idx: int
    pot: tuple[int, int]
    start_t: int
    target: list[int]  # order active when the first ingredient went in
    adds: list = field(default_factory=list)  # (t, agent, ingredient)
    consistent_sizes: list = field(default_factory=list)  # (t, |consistent set|)
    full_t: Optional[int] = None
    plated_t: Optional[int] = None
    plated_by: Optional[int] = None
    delivered_t: Optional[int] = None
    delivered_by: Optional[int] = None
    correct: Optional[bool] = None
    contents: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Environment wrapper
# --------------------------------------------------------------------------- #
class CoopEnv:
    def __init__(
        self,
        layout: str,
        recipes: list[list[int]] | str = "mixed_01",
        view_radius: int = 2,
        max_steps: int = 200,
        negative_rewards: bool = True,
        resample_on_delivery: bool = True,
        random_agent_positions: bool = False,
        ingredient_names: Optional[list[str]] = None,
    ):
        if isinstance(recipes, str):
            recipes = RECIPE_SETS[recipes]
        self.recipes = [sorted(r) for r in recipes]
        self.layout_name = layout
        self.layout = L.Layout.from_string(LAYOUT_STRINGS[layout], possible_recipes=self.recipes)
        self.view_radius = view_radius
        self.max_steps = max_steps
        self.resample_on_delivery = resample_on_delivery
        self.ingredient_names = ingredient_names or DEFAULT_INGREDIENT_NAMES
        # We handle the recipe sequence ourselves -> env never resamples.
        self.env = OvercookedV2(
            layout=self.layout,
            max_steps=max_steps,
            agent_view_size=None,
            negative_rewards=negative_rewards,
            sample_recipe_on_delivery=False,
            random_agent_positions=random_agent_positions,
        )
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
        self.plate_piles, self.pots, self.goals, self.indicators, self.buttons, self.counters = [], [], [], [], [], []
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
                elif v == StaticObject.RECIPE_INDICATOR:
                    self.indicators.append((x, y))
                elif v == StaticObject.BUTTON_RECIPE_INDICATOR:
                    self.buttons.append((x, y))
                elif v == StaticObject.WALL and any(self._walkable(n) for n in self.neighbors((x, y))):
                    self.counters.append((x, y))

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
        sym = {
            int(StaticObject.EMPTY): ".", int(StaticObject.WALL): "W", int(StaticObject.GOAL): "X",
            int(StaticObject.POT): "P", int(StaticObject.RECIPE_INDICATOR): "R",
            int(StaticObject.BUTTON_RECIPE_INDICATOR): "L", int(StaticObject.PLATE_PILE): "B",
        }
        rows = ["     x=" + "".join(str(x % 10) for x in range(self.W))]
        for y in range(self.H):
            line = ""
            for x in range(self.W):
                v = int(self.static[y, x])
                line += str(v - int(StaticObject.INGREDIENT_PILE_BASE)) if StaticObject.is_ingredient_pile(v) else sym.get(v, "?")
            rows.append(f"  y={y}  {line}")
        legend = (
            "Legend: '.' floor (walkable); W counter/wall (a W next to floor can hold one item); "
            "P pot; X serving window (deliver dishes here); B plate pile; "
            + ("R order display; " if self.indicators else "")
            + ("L order button; " if self.buttons else "")
            + "digits = ingredient piles ("
            + ", ".join(f"{i}={self.ing_name(i)}" for i in sorted(self.piles))
            + ")."
        )
        return "\n".join(rows) + "\n" + legend

    # ---------------- episode lifecycle ---------------- #
    def reset(self, seed: int, recipe_seq_len: int = 64):
        rng = np.random.default_rng(seed)
        self.recipe_seq = [self.recipes[i] for i in rng.integers(0, len(self.recipes), size=recipe_seq_len)]
        self.recipe_ptr = 0
        self.key = jax.random.PRNGKey(seed)
        with _JAX_LOCK:
            self.key, k = jax.random.split(self.key)
            _, st = self.env.reset(k)
            st = st.replace(recipe=jnp.array(encode_recipe(self.recipe_seq[0]), dtype=st.recipe.dtype))
        self.state = st
        self._sync()
        self.t = 0
        self.score = 0.0
        self.events: list[Event] = []
        self.soups: list[Soup] = []
        self._open_soup: dict = {}  # pot -> Soup still in the pot (filling / cooking / ready)
        self._plated: list = []     # soups taken out on a plate, not yet delivered
        self.observed: list[deque] = [deque(maxlen=200) for _ in range(self.num_agents)]
        self.last_seen: list[dict] = [dict() for _ in range(self.num_agents)]
        self.recipe_history = [(0, list(self.recipe_seq[0]))]
        self._update_last_seen()
        return self

    @property
    def done(self) -> bool:
        return self.t >= self.max_steps

    @property
    def current_recipe(self) -> list[int]:
        return list(self.recipe_seq[self.recipe_ptr])

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
        ax, ay = self.agent_pos(i)
        return max(abs(p[0] - ax), abs(p[1] - ay)) <= self.view_radius

    def item_text(self, obj: int) -> str:
        if obj == 0:
            return "nothing"
        ings = decode_ingredients(obj)
        if is_plate(obj) and is_cooked(obj):
            return f"a dish of {'+'.join(self.ing_name(i) for i in ings)} soup"
        if is_plate(obj):
            return "an empty plate"
        if len(ings) == 1:
            n = self.ing_name(ings[0])
            return f"{'an' if n[0] in 'aeiou' else 'a'} {n}"
        return "+".join(self.ing_name(i) for i in ings)

    def consistent_recipes(self, contents: list[int]) -> list[list[int]]:
        return [r for r in self.recipes if multiset_contains(r, contents)]

    def visible_order(self, i) -> Optional[list[int]]:
        """The current order, if agent i can read an order display (or an active button) right now."""
        for p in self.indicators:
            if self.in_view(i, p):
                return self.current_recipe
        for p in self.buttons:
            if self.in_view(i, p) and self.cell(p)[2] > 0:
                return self.current_recipe
        return None

    def _update_last_seen(self):
        for i in range(self.num_agents):
            for p in self.pots:
                if self.in_view(i, p):
                    self.last_seen[i][p] = (self.t, self.pot_status(p))

    # ---------------- stepping ---------------- #
    def step(self, actions: dict[int, int]):
        prev = self.state
        prev_inv = [self.inventory(i) for i in range(self.num_agents)]
        prev_cells = {}
        for i in range(self.num_agents):
            fx, fy = self._fwd(i)
            if 0 <= fx < self.W and 0 <= fy < self.H:
                prev_cells[i] = self.cell((fx, fy))
        prev_recipe = self.current_recipe
        acts = {f"agent_{i}": jnp.array(actions.get(i, A_STAY)) for i in range(self.num_agents)}
        with _JAX_LOCK:
            self.key, k = jax.random.split(self.key)
            _, st, rew, _, _ = self._step_fn(k, prev, acts)
        self.state = st
        self._sync()
        r = float(rew["agent_0"])
        self.score += r
        self.t += 1
        new_events = self._detect_events(actions, prev_inv, prev_cells, prev_recipe)
        # Advance our recipe sequence on correct deliveries.
        if any(e.kind == "deliver" and e.correct for e in new_events) and self.resample_on_delivery:
            self.recipe_ptr = min(self.recipe_ptr + 1, len(self.recipe_seq) - 1)
            self.state = self.state.replace(
                recipe=jnp.array(encode_recipe(self.current_recipe), dtype=self.state.recipe.dtype)
            )
            self.recipe_history.append((self.t, self.current_recipe))
        self.events.extend(new_events)
        for e in new_events:
            for j in range(self.num_agents):
                if j == e.agent or self.in_view(j, e.pos) or self.in_view(j, self.agent_pos(e.agent)):
                    self.observed[j].append(e)
        self._update_last_seen()
        return r, new_events

    def _fwd(self, i):
        x, y = self.agent_pos(i)
        dx, dy = DIR_VEC[self.agent_dir(i)]
        return x + dx, y + dy

    def _detect_events(self, actions, prev_inv, prev_cells, prev_recipe) -> list[Event]:
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
            if static == StaticObject.POT and inv0 != 0 and inv1 == 0 and not is_plate(inv0):
                added = decode_ingredients(inv0)[0]
                evs.append(Event(self.t, "pot_add", i, p, self.ing_name(added), extra={"ingredient": added}))
                self._on_pot_add(p, i, added)
            elif static == StaticObject.POT and is_plate(inv0) and is_cooked(inv1):
                evs.append(Event(self.t, "plate_soup", i, p, self.item_text(inv1)))
                # The pot is free again: the next add starts a new soup even if this
                # dish has not been delivered yet.
                s = self._open_soup.pop(p, None)
                if s:
                    s.plated_t, s.plated_by = self.t, i
                    self._plated.append(s)
            elif static == StaticObject.GOAL and is_cooked(inv0) and inv1 == 0:
                delivered = decode_ingredients(inv0)
                ok = sorted(delivered) == sorted(prev_recipe)
                evs.append(Event(self.t, "deliver", i, p, self.item_text(inv0), correct=ok))
                self._close_soup(i, delivered, ok)
            elif static == StaticObject.WALL and inv0 != 0 and inv1 == 0:
                evs.append(Event(self.t, "counter_place", i, p, self.item_text(inv0)))
            elif static == StaticObject.WALL and inv0 == 0 and inv1 != 0:
                evs.append(Event(self.t, "counter_pick", i, p, self.item_text(inv1)))
            elif StaticObject.is_ingredient_pile(static) or static == StaticObject.PLATE_PILE:
                if inv0 == 0 and inv1 != 0:
                    evs.append(Event(self.t, "fetch", i, p, self.item_text(inv1)))
            elif static == StaticObject.BUTTON_RECIPE_INDICATOR and extra > p_extra:
                evs.append(Event(self.t, "button", i, p))
        # Pot fills / cooking start bookkeeping
        for p, s in self._open_soup.items():
            status, contents, _ = self.pot_status(p)
            if s.full_t is None and len(contents) == 3:
                s.full_t = self.t
        return evs

    def _on_pot_add(self, p, agent, ingredient):
        s = self._open_soup.get(p)
        if s is None:
            s = Soup(idx=len(self.soups), pot=p, start_t=self.t, target=self.current_recipe)
            self.soups.append(s)
            self._open_soup[p] = s
        s.adds.append((self.t, agent, ingredient))
        _, contents, _ = self.pot_status(p)
        s.contents = contents
        s.consistent_sizes.append((self.t, len(self.consistent_recipes(contents))))

    def _close_soup(self, agent, delivered, ok):
        # The delivered dish is the one this agent plated most recently (dishes can also be
        # handed over via counters, so fall back to the most recent plated soup).
        cand = [s for s in self._plated if s.plated_by == agent] or self._plated
        if not cand:
            return
        s = max(cand, key=lambda z: z.plated_t)
        s.delivered_t, s.delivered_by, s.correct = self.t, agent, ok
        self._plated.remove(s)

    # ---------------- text observation ---------------- #
    def observation_text(self, i: int, oracle_recipe: bool = False, n_events: int = 12) -> str:
        x, y = self.agent_pos(i)
        lines = [
            f"Time step {self.t} of {self.max_steps}. Score so far: {self.score:+.0f}.",
            f"You are at ({x},{y}) facing {DIR_NAME[self.agent_dir(i)]}, holding {self.item_text(self.inventory(i))}.",
        ]
        order = self.visible_order(i)
        if oracle_recipe:
            lines.append(f"Current order (given to you): {self.recipe_str(self.current_recipe)} soup.")
        elif order is not None:
            lines.append(f"Order display (visible now): {self.recipe_str(order)} soup.")
        else:
            lines.append("Order display: not visible from where you are.")
        j = 1 - i if self.num_agents == 2 else None
        if j is not None:
            if self.in_view(i, self.agent_pos(j)):
                px, py = self.agent_pos(j)
                lines.append(f"Partner: at ({px},{py}) facing {DIR_NAME[self.agent_dir(j)]}, holding {self.item_text(self.inventory(j))}.")
            else:
                lines.append("Partner: not in view.")
        for p in self.pots:
            if self.in_view(i, p):
                status, contents, extra = self.pot_status(p)
                lines.append(f"Pot {xy(p)}: {self._pot_text(status, contents, extra)} (visible now).")
            elif p in self.last_seen[i]:
                ts, (status, contents, extra) = self.last_seen[i][p]
                lines.append(f"Pot {xy(p)}: last seen at t={ts}: {self._pot_text(status, contents, extra)}.")
            else:
                lines.append(f"Pot {xy(p)}: not seen yet.")
        items = []
        for c in self.counters:
            if self.in_view(i, c):
                obj = self.cell(c)[1]
                if obj:
                    items.append(f"{xy(c)}: {self.item_text(obj)}")
        lines.append("Items on visible counters: " + ("; ".join(items) if items else "none") + ".")
        ev = list(self.observed[i])[-n_events:]
        if ev:
            lines.append("Events you observed (most recent last):")
            for e in ev:
                lines.append("  " + self.event_text(e, i))
        return "\n".join(lines)

    def _pot_text(self, status, contents, extra):
        names = ", ".join(self.ing_name(c) for c in contents) or "empty"
        if status == "cooking":
            return f"cooking [{names}], {extra} steps left"
        if status == "ready":
            return f"soup ready [{names}]"
        return f"{len(contents)}/3 ingredients [{names}]"

    def event_text(self, e: Event, viewer: int) -> str:
        who = "you" if e.agent == viewer else "partner"
        d = {
            "pot_add": f"{who} put {e.item} into pot {xy(e.pos)}",
            "counter_place": f"{who} placed {e.item} on counter {xy(e.pos)}",
            "counter_pick": f"{who} picked up {e.item} from counter {xy(e.pos)}",
            "fetch": f"{who} took {e.item} from {xy(e.pos)}",
            "plate_soup": f"{who} put the soup on a plate",
            "deliver": f"{who} delivered {e.item}: {'CORRECT (+20)' if e.correct else 'WRONG order'}",
            "button": f"{who} pressed the order button {xy(e.pos)}",
        }
        return f"[t={e.t}] " + d.get(e.kind, e.kind)
