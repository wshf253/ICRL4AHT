"""Controlled two-pot scenario; not an unchanged ICRL4AHT benchmark task."""
import hashlib

import jax
import jax.numpy as jnp
import numpy as np

from envs.overcooked_v2.layouts import Layout, overcooked_v2_layouts
from envs.overcooked_v2.overcooked import OvercookedV2
from envs.overcooked_v2.common import DynamicObject

LAYOUT_TEXT = """
WWPWPWW
O     B
W A A W
W     W
WWXWXWW
"""
SUPPORTED_LAYOUTS = ('controlled', 'cramped_room', 'asymm_advantages', 'coord_ring', 'forced_coord', 'counter_circuit')

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
        raise ValueError('Unsupported layout: executor supports fixed ingredient-0 recipes only')
    if start not in ('controlled', 'empty'):
        raise ValueError(start)
    layout = (Layout.from_string(LAYOUT_TEXT, possible_recipes=[[0, 0, 0]])
              if layout_name == 'controlled' else overcooked_v2_layouts[layout_name])
    env = OvercookedV2(layout=layout, max_steps=steps,
                       agent_view_size=None if observation == 'full' else 2,
                       random_reset=False, random_agent_positions=False,
                       negative_rewards=True, sample_recipe_on_delivery=False)
    env.research_start = start
    return env


def initial_state(env, seed):
    # Initialization is independent of condition and profile. Seeds control the
    # cooking pot, ingredient count in the other pot, and initial cooking timer.
    _, state = env.reset(jax.random.PRNGKey(seed))
    if env.research_start == 'empty':
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
