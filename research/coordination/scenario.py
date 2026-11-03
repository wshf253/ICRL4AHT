"""Controlled RQ3 kitchens; not unchanged ICRL4AHT benchmark tasks.

Single-recipe layouts use one public recipe (three ingredient_0). Multi-recipe
layouts (Stage 1) draw one recipe per episode from the layout's recipe set,
keep it fixed, and show it only on recipe indicator cells ('R').
"""
import hashlib

import jax
import jax.numpy as jnp
import numpy as np

from envs.overcooked_v2.layouts import Layout, overcooked_v2_layouts
from envs.overcooked_v2.overcooked import OvercookedV2
from envs.overcooked_v2.common import DynamicObject, StaticObject

LAYOUTS = {
    # Original RQ3 pilot kitchen. Kept bit-identical (initial states included)
    # so earlier runs stay comparable.
    'two_pot_small': """
WWPWPWW
O     B
W A A W
W     W
WWXWXWW
""",
    # Ingredients on the far-left wall, plates and serving on the far-right
    # wall, three pots in between. A wrongly duplicated job costs a full
    # crossing, and three pots keep cooking time from capping throughput.
    'long_three_pot': """
WWWPWPWPWWW
0         B
0   A A   X
0         B
WWWWWWWWWWW
""",
    # Same geometry with two pots: separates "distance" from "pot count".
    'long_two_pot': """
WWWWPWPWWWW
0         B
0   A A   X
0         B
WWWWWWWWWWW
""",
}
# Stage 1 multi-recipe kitchen: long_three_pot geometry, ingredient_0 on the
# left wall, ingredient_1 mid-bottom (equidistant from both starts), recipe
# indicator between the ingredient_0 piles. Both recipes need both types, so
# every episode requires choosing the right ingredient per pot. Chosen over
# four other geometries by research.coordination.screen (RQ3_EXPERIMENT.md).
LAYOUTS['long_three_pot_mixed'] = """
WWWPWPWPWWW
0         B
R   A A   X
0         B
WWWWW1WWWWW
"""
LAYOUT_RECIPES = {'long_three_pot_mixed': [[0, 0, 1], [0, 1, 1]]}
DEFAULT_LAYOUT = 'two_pot_small'
# Backward-compatible name for the original layout text.
LAYOUT_TEXT = LAYOUTS[DEFAULT_LAYOUT]

LAYOUTS['controlled'] = LAYOUT_TEXT
SUPPORTED_LAYOUTS = ('controlled', 'cramped_room', 'asymm_advantages', 'coord_ring', 'forced_coord', 'counter_circuit', 'two_pot_small', 'long_two_pot', 'long_three_pot')
# Built-in multi-recipe layouts where each agent alone reaches every station
# and a recipe indicator (no counter hand-offs needed), plus the custom one.
MULTI_RECIPE_LAYOUTS = ('long_three_pot_mixed', 'cramped_room_up', 'cramped_room_down',
                        'asymm_advantages_recipes_both')
SUPPORTED_LAYOUTS += MULTI_RECIPE_LAYOUTS

PROFILES = ('prep', 'delivery')
CONDITIONS = ('none', 'correct', 'opposite')
DESCRIPTIONS = {
    'prep': 'When both ingredient preparation and plating/delivery jobs are available, '
            'this partner prioritizes ingredient preparation. It can do both types of work. '
            'It commits to a job until completion or timeout, then chooses again.',
    'delivery': 'When both ingredient preparation and plating/delivery jobs are available, '
                'this partner prioritizes plating/delivery. It can do both types of work. '
                'It commits to a job until completion or timeout, then chooses again.',
}


def partner_information(profile, condition):
    if condition == 'none':
        return 'No prior information about this partner is available.'
    if condition == 'opposite':
        profile = 'delivery' if profile == 'prep' else 'prep'
    elif condition != 'correct':
        raise ValueError(condition)
    return DESCRIPTIONS[profile]


def create_env(steps=200, observation='full', layout_name='controlled', start='controlled'):
    if observation not in ('full', 'partial'):
        raise ValueError(observation)
    if layout_name not in SUPPORTED_LAYOUTS:
        raise ValueError(f'Unsupported layout {layout_name!r}: the executor needs open kitchens without '
                         f'counter hand-offs; choose from {SUPPORTED_LAYOUTS}')
    if start not in ('controlled', 'empty'):
        raise ValueError(start)
    layout = (Layout.from_string(LAYOUTS[layout_name],
                                 possible_recipes=LAYOUT_RECIPES.get(layout_name, [[0, 0, 0]]))
              if layout_name in LAYOUTS else overcooked_v2_layouts[layout_name])
    env = OvercookedV2(layout=layout, max_steps=steps,
                       agent_view_size=None if observation == 'full' else 2,
                       random_reset=False, random_agent_positions=False,
                       negative_rewards=True, sample_recipe_on_delivery=False)
    env.research_start = start
    env.rq3_layout = layout_name
    return env


def pot_positions(static_map):
    """Pots as (x, y), ordered left to right then top to bottom."""
    ys, xs = np.nonzero(np.asarray(static_map) == int(StaticObject.POT))
    return sorted(zip(xs.tolist(), ys.tolist()))


def is_multi_recipe(env):
    return len(env.layout.possible_recipes) > 1 or env.layout.num_ingredients > 1


def encode_ingredients(items):
    return sum(int(DynamicObject.ingredient(int(i))) for i in items)


def _multi_recipe_initial_state(env, seed, state):
    # The recipe is chosen by seed (seed mod number of recipes) so recipes are
    # balanced across seeds; it then stays fixed (sample_recipe_on_delivery off).
    # Controlled start: one pot already cooks a correct soup; every other pot
    # holds 0-2 ingredients drawn from the recipe, so a needed prep job exists.
    possible = env.layout.possible_recipes
    recipe = [int(i) for i in possible[seed % len(possible)]]
    state = state.replace(recipe=jnp.asarray(encode_ingredients(recipe), dtype=state.recipe.dtype))
    if env.research_start == 'empty':
        return env.get_obs(state), state
    rng = np.random.default_rng(seed)
    pots = pot_positions(env.layout.static_objects)
    if not pots:
        raise ValueError('Layout needs at least one pot')
    cooking = pots[int(rng.choice(len(pots)))]
    timer = int(rng.integers(5, 11))
    grid = state.grid.at[cooking[1], cooking[0], 1].set(encode_ingredients(recipe))
    grid = grid.at[cooking[1], cooking[0], 2].set(timer)
    for x, y in pots:
        if (x, y) != cooking:
            items = rng.permutation(recipe)[:int(rng.integers(0, 3))]
            grid = grid.at[y, x, 1].set(encode_ingredients(items))
    state = state.replace(grid=grid)
    return env.get_obs(state), state


def initial_state(env, seed):
    # Initialization is independent of condition and profile. Seeds control the
    # cooking pot, ingredient count in the other pot, and initial cooking timer.
    _, state = env.reset(jax.random.PRNGKey(seed))
    if is_multi_recipe(env):
        return _multi_recipe_initial_state(env, seed, state)
    if env.research_start == 'empty':
        return env.get_obs(state), state
    if env.rq3_layout in ('long_two_pot', 'long_three_pot'):
        rng = np.random.default_rng(seed)
        pots = pot_positions(env.layout.static_objects)
        cooking = pots[int(rng.choice(len(pots)))]
        timer = int(rng.integers(5, 11))
        ing = int(DynamicObject.ingredient(0))
        grid = state.grid.at[cooking[1], cooking[0], 1].set(3 * ing)
        grid = grid.at[cooking[1], cooking[0], 2].set(timer)
        for x, y in pots:
            if (x, y) != cooking:
                grid = grid.at[y, x, 1].set(int(rng.integers(0, 3)) * ing)
        state = state.replace(grid=grid)
        return env.get_obs(state), state
    rng = np.random.default_rng(seed)
    pots = np.argwhere(np.asarray(env.layout.static_objects) == 5)
    if not len(pots):
        raise ValueError('Layout needs at least one pot')
    index = int(rng.choice(len(pots)))
    y, x = map(int, pots[index])
    timer = int(rng.integers(5, 11))
    count = int(rng.integers(0, 3))
    ing = int(DynamicObject.ingredient(0))
    grid = state.grid.at[y, x, 1].set(3 * ing)
    grid = grid.at[y, x, 2].set(timer)
    if len(pots) > 1:
        py, px = map(int, pots[(index + 1) % len(pots)])
        grid = grid.at[py, px, 1].set(count * ing)
    state = state.replace(grid=grid)
    return env.get_obs(state), state


def state_hash(state):
    h = hashlib.sha256()
    for leaf in jax.tree.leaves(state):
        a = np.asarray(leaf)
        h.update(str((a.shape, a.dtype)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def primitive_key(seed, step):
    return jax.random.fold_in(jax.random.PRNGKey(seed), step)
