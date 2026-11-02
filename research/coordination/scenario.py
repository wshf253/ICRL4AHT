"""Controlled RQ3 kitchens; not unchanged ICRL4AHT benchmark tasks.

Each layout is an open kitchen with one fixed public recipe (three of ingredient
0), so the executor never needs counter hand-offs or recipe inference. Layouts
differ only in geometry: how far apart ingredient, pot and plate/serve stations
are, and how many pots run in parallel. See RQ3_EXPERIMENT.md and
``research.coordination.screen`` for how a layout is checked before LLM runs.
"""
import hashlib

import jax
import numpy as np

from envs.overcooked_v2.layouts import Layout
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
DEFAULT_LAYOUT = 'two_pot_small'
# Backward-compatible name for the original layout text.
LAYOUT_TEXT = LAYOUTS[DEFAULT_LAYOUT]

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


def create_env(steps=200, observation='full', layout=DEFAULT_LAYOUT):
    if observation not in ('full', 'partial'):
        raise ValueError(observation)
    if layout not in LAYOUTS:
        raise ValueError(f'Unknown layout {layout!r}; choose from {sorted(LAYOUTS)}')
    env = OvercookedV2(layout=Layout.from_string(LAYOUTS[layout], possible_recipes=[[0, 0, 0]]),
                       max_steps=steps,
                       agent_view_size=None if observation == 'full' else 2,
                       random_reset=False, random_agent_positions=False,
                       negative_rewards=True, sample_recipe_on_delivery=False)
    env.rq3_layout = layout
    return env


def pot_positions(static_map):
    """Pots as (x, y), ordered left to right then top to bottom."""
    ys, xs = np.nonzero(np.asarray(static_map) == int(StaticObject.POT))
    return sorted(zip(xs.tolist(), ys.tolist()))


def initial_state(env, seed):
    # Initialization is independent of condition and profile. The seed picks
    # which pot is already cooking and its remaining time, and how many
    # ingredients each other pot holds (0-2, so a prep job always exists).
    # For two_pot_small this consumes the RNG exactly as the original pilot.
    _, state = env.reset(jax.random.PRNGKey(seed))
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


def state_hash(state):
    h = hashlib.sha256()
    for leaf in jax.tree.leaves(state):
        a = np.asarray(leaf)
        h.update(str((a.shape, a.dtype)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def primitive_key(seed, step):
    return jax.random.fold_in(jax.random.PRNGKey(seed), step)
