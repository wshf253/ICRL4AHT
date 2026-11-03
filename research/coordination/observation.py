"""Decode DEFAULT observations, with explicitly public map/recipe/proprioception.

This module never receives the simulator State. Unseen dynamic cells are unknown,
not empty. The map is public. With one possible recipe the recipe is public too;
with several, an agent only knows the recipe after observing a recipe indicator.
"""
import numpy as np
from envs.overcooked_v2.common import StaticObject


def unpack_inventory(bits, n):
    return {'plate': bool(bits[0]), 'cooked': bool(bits[1]),
            'ingredients': [int(v) for v in bits[2:2+n]]}


def inventory_kind(inv):
    if inv['plate'] and inv['cooked']:
        return 'dish'
    if inv['plate']:
        return 'plate'
    if sum(inv['ingredients']):
        return 'ingredient'
    return 'empty'


def decode_observation(obs, static_map, pose, n=1, view_size=None):
    a = np.asarray(obs)
    static_map = np.asarray(static_map)
    h, w = static_map.shape
    block = 7 + n  # position, four directions, plate/cooked/n ingredients
    dyn_offset = 2 * block + 6 + n
    recipe_offset = dyn_offset + 2 + n  # nonzero only on visible recipe indicators
    timer_offset = dyn_offset + 2 * (2 + n)
    expected = timer_offset + 1
    if a.shape[-1] != expected:
        raise ValueError(f'Expected default observation with {expected} channels, got {a.shape}')
    ox, oy = (0, 0) if view_size is None else (pose[0]-view_size, pose[1]-view_size)
    cells, self_info, others, recipe = {}, None, [], None
    for ly in range(a.shape[0]):
        for lx in range(a.shape[1]):
            x, y = ox+lx, oy+ly
            if not (0 <= x < w and 0 <= y < h):
                continue
            cell = a[ly, lx]
            cells[(x, y)] = {
                **unpack_inventory(cell[dyn_offset:dyn_offset+2+n], n),
                'timer': int(cell[timer_offset]),
            }
            shown = [int(v) for v in cell[recipe_offset+2:recipe_offset+2+n]]
            if sum(shown):
                recipe = shown
            for offset, is_self in ((0, True), (block, False)):
                if cell[offset] > 0:
                    agent = {'pos': [x, y], 'direction': int(np.argmax(cell[offset+1:offset+5])),
                             'inventory': unpack_inventory(cell[offset+5:offset+7+n], n)}
                    if is_self:
                        self_info = agent
                    else:
                        others.append(agent)
    if self_info is None or self_info['pos'] != list(pose):
        raise ValueError('Self pose does not match observation')
    return {'self': self_info, 'others': others, 'cells': cells, 'recipe': recipe}


def recipe_counts(recipe, n):
    """[0, 0, 1] -> [2, 1] for n=2 ingredient types."""
    return [sum(int(r) == i for r in recipe) for i in range(n)]


class RecipeTracker:
    """What one agent knows about the current recipe, from its own observations.

    Stage 1 keeps the recipe fixed for the whole episode, so a recipe seen once
    stays valid. A layout with a single possible recipe makes it public.
    """

    def __init__(self, possible_recipes, n):
        self.n = n
        self.possible = [recipe_counts(r, n) for r in possible_recipes]
        self.public = self.possible[0] if len(self.possible) == 1 else None
        self.memory = None

    @property
    def multi(self):
        return self.public is None

    def update(self, view):
        """Annotate view['known_recipe'] (counts or None); return planner info."""
        if self.public is not None:
            view['known_recipe'] = self.public
            return None  # public recipe: planner input stays in the original schema
        if view['recipe'] is not None:
            self.memory, source = view['recipe'], 'observed_now'
        else:
            source = 'remembered' if self.memory is not None else 'unknown'
        view['known_recipe'] = self.memory
        return {'ingredient_counts': None if self.memory is None else
                {f'ingredient_{i}': c for i, c in enumerate(self.memory)},
                'source': source, 'fixed_for_episode': True,
                'possible_recipes': [{f'ingredient_{i}': c for i, c in enumerate(r)}
                                     for r in self.possible]}


def ingredient_counts(inv):
    return {f'ingredient_{i}': int(count) for i, count in enumerate(inv['ingredients'])}


def planner_agent(agent):
    inv = agent['inventory']
    kind = inventory_kind(inv)
    holding = {'empty': 'empty', 'plate': 'empty_plate', 'dish': 'cooked_soup_on_plate'}.get(kind)
    if kind == 'ingredient':
        holding = next(f'ingredient_{i}' for i, count in enumerate(inv['ingredients']) if count)
    return {'pos': list(agent['pos']),
            'facing': ('up', 'down', 'right', 'left')[agent['direction']],
            'holding': holding, 'ingredient_counts': ingredient_counts(inv)}


def planner_contents(cell, kind):
    if cell is None:
        return None
    counts = ingredient_counts(cell)
    if kind == 'pot':
        status = ('ready' if cell['cooked'] else 'cooking' if cell['timer'] > 0
                  else 'empty' if sum(counts.values()) == 0 else 'filling')
        return {'status': status, 'ingredient_counts': counts,
                'remaining_cook_steps': int(cell['timer']) if status == 'cooking' else 0}
    # Supply and delivery stations are static facilities, not held inventories.
    return {'station_type': kind}


def planner_view(view, static_map, recipe_info=None):
    """Explicit counts and object names; unseen dynamics remain unknown.

    recipe_info is None for single-recipe layouts, which keeps their planner
    input byte-identical to earlier runs (public_recipe key).
    """
    stations = []
    names = {int(StaticObject.POT): 'pot', int(StaticObject.PLATE_PILE): 'plates',
             int(StaticObject.GOAL): 'serve', int(StaticObject.RECIPE_INDICATOR): 'recipe_indicator'}
    for y, row in enumerate(static_map):
        for x, raw in enumerate(row):
            obj = int(raw)
            kind = f'ingredient_{obj-10}' if obj >= 10 else names.get(obj)
            if kind:
                stations.append({'kind': kind, 'pos': [x, y],
                                 'observed_now': (x, y) in view['cells'],
                                 'contents': planner_contents(view['cells'].get((x, y)), kind)})
    result = {'schema_version': 2, 'self': planner_agent(view['self']),
              'visible_partner': [planner_agent(a) for a in view['others']],
              'stations': stations, 'public_recipe': {'ingredient_counts': {'ingredient_0': 3}},
              'unseen_dynamic_state': 'unknown'}
    if recipe_info is not None:
        del result['public_recipe']
        result['schema_version'] = 3
        result['recipe'] = recipe_info
    return result
