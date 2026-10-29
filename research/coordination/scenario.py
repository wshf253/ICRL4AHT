"""Controlled two-pot scenario; not an unchanged ICRL4AHT benchmark task."""
import hashlib

import jax
import jax.numpy as jnp
import numpy as np

from envs.overcooked_v2.layouts import Layout
from envs.overcooked_v2.overcooked import OvercookedV2
from envs.overcooked_v2.common import DynamicObject

LAYOUT_TEXT = """
WWPWPWW
O     B
W A A W
W     W
WWXWXWW
"""
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


def create_env(steps=200, observation='full'):
    if observation not in ('full', 'partial'):
        raise ValueError(observation)
    layout = Layout.from_string(LAYOUT_TEXT, possible_recipes=[[0, 0, 0]])
    return OvercookedV2(layout=layout, max_steps=steps,
                       agent_view_size=None if observation == 'full' else 2,
                       random_reset=False, random_agent_positions=False,
                       negative_rewards=True, sample_recipe_on_delivery=False)


def initial_state(env, seed):
    # Initialization is independent of condition and profile. Seeds control the
    # cooking pot, ingredient count in the other pot, and initial cooking timer.
    _, state = env.reset(jax.random.PRNGKey(seed))
    rng = np.random.default_rng(seed)
    cooking_x = int(rng.choice([2, 4]))
    prep_x = 6 - cooking_x
    timer = int(rng.integers(5, 11))
    count = int(rng.integers(0, 3))
    ing = int(DynamicObject.ingredient(0))
    grid = state.grid.at[0, cooking_x, 1].set(3 * ing)
    grid = grid.at[0, cooking_x, 2].set(timer)
    grid = grid.at[0, prep_x, 1].set(count * ing)
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
