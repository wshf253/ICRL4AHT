"""Decode DEFAULT observations, with explicitly public map/recipe/proprioception.

This module never receives the simulator State. Unseen dynamic cells are unknown,
not empty. The fixed map and recipe are public in this controlled pilot.
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
    timer_offset = dyn_offset + 2 * (2 + n)
    expected = timer_offset + 1
    if a.shape[-1] != expected:
        raise ValueError(f'Expected default observation with {expected} channels, got {a.shape}')
    ox, oy = (0, 0) if view_size is None else (pose[0]-view_size, pose[1]-view_size)
    cells, self_info, others = {}, None, []
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
    return {'self': self_info, 'others': others, 'cells': cells}


def planner_view(view, static_map):
    """JSON view only: no hidden state, profile label, or partner controller state."""
    stations = []
    names = {int(StaticObject.POT): 'pot', int(StaticObject.PLATE_PILE): 'plates',
             int(StaticObject.GOAL): 'serve'}
    for y, row in enumerate(static_map):
        for x, raw in enumerate(row):
            obj = int(raw)
            kind = 'ingredient_0' if obj == 10 else names.get(obj)
            if kind:
                stations.append({'kind': kind, 'pos': [x, y],
                                 'observed_now': (x, y) in view['cells'],
                                 'contents': view['cells'].get((x, y))})
    return {'self': view['self'], 'visible_partner': view['others'],
            'stations': stations, 'public_recipe': [0, 0, 0],
            'unseen_dynamic_state': 'unknown'}
