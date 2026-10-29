"""Shared deterministic navigation/interaction skills for both agents.

No full simulator state is accepted. Decisions use decoded observations. The
partner's preference affects only task choice, never navigation or speed.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from envs.overcooked_v2.common import Actions
from .observation import inventory_kind

MOVES = [(0, -1, int(Actions.up), 0), (0, 1, int(Actions.down), 1),
         (1, 0, int(Actions.right), 2), (-1, 0, int(Actions.left), 3)]


def locations(static, value):
    return [(int(x), int(y)) for y, x in np.argwhere(static == value)]


def route(static, start, target, blocked=()):
    """Shortest path to a station's adjacent floor cell; coordinates are x,y."""
    h, w = static.shape
    start, target = tuple(start), tuple(target)
    blocked = set(map(tuple, blocked)) - {start}
    queue = deque([(start, [])])
    seen = {start}
    while queue:
        (x, y), path = queue.popleft()
        if abs(x-target[0])+abs(y-target[1]) == 1:
            return path
        for dx, dy, action, _ in MOVES:
            nxt = (x+dx, y+dy)
            if (0 <= nxt[0] < w and 0 <= nxt[1] < h and static[nxt[1], nxt[0]] == 0
                    and nxt not in blocked and nxt not in seen):
                seen.add(nxt)
                queue.append((nxt, path+[action]))
    return None


def nearest(static, start, targets, blocked=()):
    options = [(len(path), y, x) for x, y in targets
               if (path := route(static, start, (x, y), blocked)) is not None]
    if not options:
        return None
    _, y, x = min(options)
    return (x, y)


def candidates(view, static):
    kind = inventory_kind(view['self']['inventory'])
    result = []
    if kind == 'dish':
        result.append({'id': 'deliver', 'kind': 'delivery', 'target': None,
                       'description': 'Deliver the dish you are carrying.'})
    else:
        for pot in locations(static, 5):
            c = view['cells'].get(pot)
            if c is None:
                continue
            if kind in ('empty', 'ingredient') and not c['cooked'] and c['timer'] == 0 and sum(c['ingredients']) < 3:
                result.append({'id': f'prep_{pot[0]}_{pot[1]}', 'kind': 'prep', 'target': list(pot),
                               'description': 'Fetch one ingredient if needed, then add it to this pot.'})
            if kind in ('empty', 'plate') and (c['cooked'] or c['timer'] > 0):
                result.append({'id': f'serve_{pot[0]}_{pot[1]}', 'kind': 'delivery', 'target': list(pot),
                               'description': 'Fetch a plate if needed, collect this soup when ready, then deliver it.'})
        if kind in ('plate', 'ingredient'):
            result.append({'id': 'put_down', 'kind': 'recovery', 'target': None,
                           'description': 'Put the carried item onto an observed empty counter to free your hands.'})
    # Inspect known pots in partial observation; no hidden dynamics are consulted.
    for pot in locations(static, 5):
        if pot not in view['cells']:
            result.append({'id': f'inspect_{pot[0]}_{pot[1]}', 'kind': 'inspect', 'target': list(pot),
                           'description': 'Move near this pot to observe its current state.'})
    result.append({'id': 'wait', 'kind': 'wait', 'target': None, 'description': 'Wait briefly.'})
    return result


def preferred_task(menu, profile, pose, static):
    work = [c for c in menu if c['kind'] in ('prep', 'delivery')]
    preferred = [c for c in work if c['kind'] == profile]
    pool = preferred or work or [c for c in menu if c['kind'] in ('recovery', 'inspect')]
    if not pool:
        return next(c for c in menu if c['id'] == 'wait')
    def cost(c):
        if c['target'] is None:
            return (0, c['id'])
        path = route(static, pose, c['target'])
        return (len(path) if path is not None else 10000, c['id'])
    return min(pool, key=cost)


@dataclass
class Job:
    task: dict
    age: int = 0
    previous_kind: str = 'empty'
    interacted: bool = False
    attempted: str = ''
    blocked_steps: int = 0


class Executor:
    def __init__(self, static_map, agent_id, max_job_steps=32):
        self.static = np.asarray(static_map)
        self.agent_id = agent_id
        self.max_job_steps = max_job_steps
        self.job = None
        self.last_outcome = None

    def assign(self, task, view):
        self.job = Job(dict(task), previous_kind=inventory_kind(view['self']['inventory']))
        self.last_outcome = None

    def observe(self, view):
        """Complete jobs only from own inventory/observed changes, never reward oracle."""
        j = self.job
        if j is None:
            return
        kind = inventory_kind(view['self']['inventory'])
        completed = j.interacted and (
            (j.attempted == 'add' and j.previous_kind == 'ingredient' and kind == 'empty') or
            (j.attempted == 'deliver' and j.previous_kind == 'dish' and kind == 'empty') or
            (j.attempted == 'put_down' and j.previous_kind != 'empty' and kind == 'empty'))
        c = view['cells'].get(tuple(j.task['target'])) if j.task['target'] else None
        invalid = (j.task['kind'] == 'prep' and c is not None and
                   (c['cooked'] or c['timer'] > 0 or sum(c['ingredients']) >= 3))
        inspect_done = j.task['kind'] == 'inspect' and c is not None
        wait_done = j.task['kind'] == 'wait' and j.age >= 2
        if completed or inspect_done or wait_done or invalid or j.age >= self.max_job_steps:
            self.last_outcome = {'task': j.task['id'], 'outcome':
                                 'completed' if completed or inspect_done or wait_done else
                                 'target_changed' if invalid else 'timeout', 'duration': j.age}
            self.job = None

    def _approach(self, target, view, interact=True):
        if target is None:
            return int(Actions.stay)
        me = view['self']
        x, y = me['pos']
        dx, dy = target[0]-x, target[1]-y
        if abs(dx)+abs(dy) == 1:
            for mx, my, act, direction in MOVES:
                if (mx, my) == (dx, dy):
                    if me['direction'] != direction:
                        return act
                    return int(Actions.interact if interact else Actions.stay)
        blocked = [o['pos'] for o in view['others']]
        path = route(self.static, (x, y), target, blocked)
        if path:
            self.job.blocked_steps = 0
            return path[0]
        self.job.blocked_steps += 1
        # Deterministic local yield for a visible blockage; identical ability for
        # both profiles, asymmetric by ID only to avoid simultaneous swapping.
        if self.job.blocked_steps >= 2 and self.job.age % 2 == self.agent_id % 2:
            for mx, my, act, _ in MOVES:
                nx, ny = x+mx, y+my
                if 0 <= ny < self.static.shape[0] and 0 <= nx < self.static.shape[1]:
                    if self.static[ny, nx] == 0 and [nx, ny] not in blocked:
                        return act
        return int(Actions.stay)

    def action(self, view):
        j = self.job
        if j is None:
            raise RuntimeError('Assign a job before requesting an action')
        me = view['self']
        inv = inventory_kind(me['inventory'])
        pos = me['pos']
        task = j.task
        target, interact, stage = None, True, ''
        if inv == 'dish':
            target = nearest(self.static, pos, locations(self.static, 4))
            stage = 'deliver'
        elif task['kind'] == 'prep':
            target = nearest(self.static, pos, locations(self.static, 10)) if inv == 'empty' else tuple(task['target'])
            stage = 'fetch' if inv == 'empty' else 'add'
        elif task['kind'] == 'delivery':
            if inv == 'empty':
                target = nearest(self.static, pos, locations(self.static, 9))
                stage = 'plate'
            elif inv == 'plate':
                target = tuple(task['target'])
                c = view['cells'].get(target)
                interact = c is not None and c['cooked']
                stage = 'collect'
                # If another agent already collected it, release the job instead
                # of carrying a plate and waiting for the whole timeout.
                if c is not None and not c['cooked'] and c['timer'] == 0:
                    j.age = self.max_job_steps
        elif task['kind'] == 'inspect':
            target, interact = tuple(task['target']), False
        elif task['kind'] == 'recovery':
            targets = [p for p in locations(self.static, 1) if p in view['cells']
                       and inventory_kind(view['cells'][p]) == 'empty']
            target = nearest(self.static, pos, targets)
            stage = 'put_down'
        action = self._approach(target, view, interact) if target is not None else int(Actions.stay)
        j.age += 1
        j.previous_kind = inv
        j.interacted = action == int(Actions.interact)
        j.attempted = stage
        return action
