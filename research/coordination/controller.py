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


def held_ingredient(inv):
    return next((i for i, count in enumerate(inv['ingredients']) if count), None)


def matches(contents, recipe):
    return recipe is not None and list(contents) == list(recipe)


def still_possible(contents, recipe):
    """Pot contents can still become exactly the recipe."""
    return recipe is not None and all(c <= r for c, r in zip(contents, recipe))


def candidates(view, static):
    """Jobs feasible from the agent's own observation.

    With one ingredient type the menu is unchanged from the single-recipe pilot.
    With several, prep jobs name the ingredient and every type is offered: the
    planner, not the menu, must work out what each pot still needs. Likewise a
    discard job is offered for every cooking/ready pot, so the menu never
    reveals which soup is wrong; discarding a correct soup is a logged mistake.
    """
    inv = view['self']['inventory']
    kind = inventory_kind(inv)
    n = len(inv['ingredients'])
    recipe = view.get('known_recipe')
    result = []
    if kind == 'dish':
        result.append({'id': 'deliver', 'kind': 'delivery', 'target': None,
                       'description': 'Deliver the dish you are carrying.'})
        if n > 1:
            result.append({'id': 'put_down', 'kind': 'recovery', 'target': None,
                           'description': 'Put the carried soup onto an observed empty counter instead of delivering it.'})
    else:
        for pot in locations(static, 5):
            c = view['cells'].get(pot)
            if c is None:
                continue
            if kind in ('empty', 'ingredient') and not c['cooked'] and c['timer'] == 0 and sum(c['ingredients']) < 3:
                if n == 1:
                    result.append({'id': f'prep_{pot[0]}_{pot[1]}', 'kind': 'prep', 'target': list(pot),
                                   'description': 'Fetch one ingredient if needed, then add it to this pot.'})
                else:
                    held = held_ingredient(inv)
                    for i in range(n) if held is None else [held]:
                        result.append({'id': f'prep_{pot[0]}_{pot[1]}_ing{i}', 'kind': 'prep',
                                       'target': list(pot), 'ingredient': i,
                                       'description': f'Fetch one ingredient_{i} if needed, then add it to this pot.'})
            if kind in ('empty', 'plate') and (c['cooked'] or c['timer'] > 0):
                result.append({'id': f'serve_{pot[0]}_{pot[1]}', 'kind': 'delivery', 'target': list(pot),
                               'description': 'Fetch a plate if needed, collect this soup when ready, then deliver it.'})
                if n > 1:
                    result.append({'id': f'discard_{pot[0]}_{pot[1]}', 'kind': 'recovery', 'target': list(pot),
                                   'description': 'Fetch a plate if needed, collect this soup when ready, then put it '
                                                  'on an empty counter instead of delivering it.'})
        if kind in ('plate', 'ingredient'):
            result.append({'id': 'put_down', 'kind': 'recovery', 'target': None,
                           'description': 'Put the carried item onto an observed empty counter to free your hands.'})
    # Inspect known pots in partial observation; no hidden dynamics are consulted.
    for pot in locations(static, 5):
        if pot not in view['cells']:
            result.append({'id': f'inspect_{pot[0]}_{pot[1]}', 'kind': 'inspect', 'target': list(pot),
                           'description': 'Move near this pot to observe its current state.'})
    if n > 1 and recipe is None:
        for x, y in locations(static, 6):
            result.append({'id': f'inspect_recipe_{x}_{y}', 'kind': 'inspect', 'target': [x, y],
                           'description': 'Move near this recipe indicator to read the current recipe.'})
    result.append({'id': 'wait', 'kind': 'wait', 'target': None, 'description': 'Wait briefly.'})
    return result


def useful_work(menu, view):
    """Scripted agents' recipe-aware filter (multi-ingredient layouts only).

    Returns (prep, plating, last_resort): prep jobs whose ingredient the target
    pot still needs; correct serves, discards and deliveries of plated soup
    (put_down when the carried soup is wrong); and prep for pots that can no
    longer match, filled only so they cook and can be discarded. Needs are
    computed per pot, never summed across pots.
    """
    recipe = view.get('known_recipe')
    cells = view['cells']
    dish = inventory_kind(view['self']['inventory']) == 'dish'
    prep, plating, last_resort = [], [], []
    for c in menu:
        pot = cells.get(tuple(c['target'])) if c['target'] else None
        if c['kind'] == 'prep' and pot is not None:
            after = list(pot['ingredients'])
            after[c['ingredient']] += 1
            if still_possible(after, recipe):
                prep.append(c)
            elif not still_possible(pot['ingredients'], recipe):
                last_resort.append(c)
        elif c['id'].startswith('serve_') and pot is not None and matches(pot['ingredients'], recipe):
            plating.append(c)
        elif c['id'].startswith('discard_') and pot is not None and not matches(pot['ingredients'], recipe):
            plating.append(c)
        elif dish and c['id'] == ('deliver' if matches(view['self']['inventory']['ingredients'], recipe)
                                   else 'put_down'):
            plating.append(c)
    return prep, plating, last_resort


def preferred_task(menu, profile, pose, static, view=None):
    """Scripted priority rule. Pass the agent's view on multi-ingredient layouts
    so it never adds an ingredient a pot does not need or delivers a wrong soup."""
    work = [c for c in menu if c['kind'] in ('prep', 'delivery')]
    preferred = [c for c in work if c['kind'] == profile]
    pool = preferred or work or [c for c in menu if c['kind'] in ('recovery', 'inspect')]
    if view is not None and len(view['self']['inventory']['ingredients']) > 1:
        if view.get('known_recipe') is None:
            pool = ([c for c in menu if c['id'].startswith('inspect_recipe')] or
                    [c for c in menu if c['kind'] == 'inspect'] or
                    [c for c in menu if c['id'] == 'put_down'])
        else:
            prep, plating, last_resort = useful_work(menu, view)
            mine = prep if profile == 'prep' else plating
            pool = (mine or prep + plating or last_resort or
                    [c for c in menu if c['id'] == 'put_down' or c['kind'] == 'inspect'])
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
    needed_at_assign: bool = False  # multi-ingredient prep: pot needed this ingredient


def ingredient_still_needed(task, view):
    """Adding task's ingredient keeps the observed pot able to match the recipe."""
    pot = view['cells'].get(tuple(task['target']))
    if pot is None or 'ingredient' not in task:
        return None  # unobserved pot or single-ingredient prep: no judgement
    after = list(pot['ingredients'])
    after[task['ingredient']] += 1
    return still_possible(after, view.get('known_recipe'))


class Executor:
    def __init__(self, static_map, agent_id, max_job_steps=32, seed=0):
        self.static = np.asarray(static_map)
        self.agent_id = agent_id
        self.max_job_steps = max_job_steps
        self.job = None
        self.last_outcome = None
        # Seeded randomness only breaks symmetric navigation conflicts. Both
        # agents use the same rule; it never depends on profile or condition.
        self.rng = np.random.default_rng([seed, agent_id])
        self.pending_move = None  # (position, cell) of the last move into free floor
        self.trail = deque(maxlen=3)

    def assign(self, task, view):
        self.job = Job(dict(task), previous_kind=inventory_kind(view['self']['inventory']))
        if task['kind'] == 'prep':
            self.job.needed_at_assign = bool(ingredient_still_needed(task, view))
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
        # Multi-ingredient: a needed ingredient became redundant (someone else
        # added it first). A choice that was wrong from the start is executed
        # as chosen, so planner mistakes remain visible in the logs.
        invalid |= (j.task['kind'] == 'prep' and j.needed_at_assign
                    and ingredient_still_needed(j.task, view) is False)
        inspect_done = j.task['kind'] == 'inspect' and c is not None
        wait_done = j.task['kind'] == 'wait' and j.age >= 2
        if completed or inspect_done or wait_done or invalid or j.age >= self.max_job_steps:
            self.last_outcome = {'task': j.task['id'], 'outcome':
                                 'completed' if completed or inspect_done or wait_done else
                                 'target_changed' if invalid else 'timeout', 'duration': j.age}
            self.job = None

    def _approach(self, target, view, interact=True):
        if target is None:
            self.pending_move = None
            return int(Actions.stay)
        me = view['self']
        x, y = me['pos']
        pos = (x, y)
        bumped = self.pending_move is not None and self.pending_move[0] == pos
        self.pending_move = None
        dx, dy = target[0]-x, target[1]-y
        if abs(dx)+abs(dy) == 1:
            self.trail.clear()  # reached the station: navigation made progress
            for mx, my, act, direction in MOVES:
                if (mx, my) == (dx, dy):
                    if me['direction'] != direction:
                        return act
                    return int(Actions.interact if interact else Actions.stay)
        blocked = {tuple(o['pos']) for o in view['others']}
        # Two identical deterministic planners in a symmetric situation repeat the
        # same conflict forever: both step into the same free cell (the env cancels
        # both moves), or both detour in mirror image (A-B-A). On such evidence,
        # back off with probability 1/2 so exactly one agent proceeds in expectation.
        self.trail.append(pos)
        revisit = len(self.trail) == 3 and self.trail[0] == pos != self.trail[1]
        path = route(self.static, pos, target, blocked)
        if path:
            if (bumped or revisit) and self.rng.random() < 0.5:
                return int(Actions.stay)
            return self._move(pos, path[0])
        # No path: the partner occupies the target's only access cell or a choke
        # point. If both wait for each other (e.g. each stands on the other's
        # access cell), a random side-step from either one releases the deadlock.
        if self.rng.random() < 0.5:
            free = [act for mx, my, act, _ in MOVES
                    if self._floor(x+mx, y+my) and (x+mx, y+my) not in blocked]
            if free:
                return self._move(pos, int(self.rng.choice(free)))
        return int(Actions.stay)

    def _floor(self, x, y):
        h, w = self.static.shape
        return 0 <= x < w and 0 <= y < h and self.static[y, x] == 0

    def _move(self, pos, action):
        dx, dy = next((mx, my) for mx, my, act, _ in MOVES if act == action)
        self.pending_move = (pos, (pos[0]+dx, pos[1]+dy))
        return action

    def _plate_and_collect(self, task, view, inv):
        target, interact, stage = None, True, ''
        if inv == 'empty':
            target = nearest(self.static, view['self']['pos'], locations(self.static, 9))
            stage = 'plate'
        elif inv == 'plate':
            target = tuple(task['target'])
            c = view['cells'].get(target)
            interact = c is not None and c['cooked']
            stage = 'collect'
            # If another agent already collected it, release the job instead
            # of carrying a plate and waiting for the whole timeout.
            if c is not None and not c['cooked'] and c['timer'] == 0:
                self.job.age = self.max_job_steps
        return target, interact, stage

    def action(self, view):
        j = self.job
        if j is None:
            raise RuntimeError('Assign a job before requesting an action')
        me = view['self']
        inv = inventory_kind(me['inventory'])
        pos = me['pos']
        task = j.task
        target, interact, stage = None, True, ''
        if inv == 'dish' and task['kind'] != 'recovery':
            target = nearest(self.static, pos, locations(self.static, 4))
            stage = 'deliver'
        elif task['kind'] == 'prep':
            pile = 10 + task.get('ingredient', 0)
            target = nearest(self.static, pos, locations(self.static, pile)) if inv == 'empty' else tuple(task['target'])
            stage = 'fetch' if inv == 'empty' else 'add'
        elif task['id'].startswith('discard_') and inv in ('empty', 'plate'):
            # Same plate/collect stages as a delivery; the dish is then put on a
            # counter by the recovery branch below instead of being served.
            target, interact, stage = self._plate_and_collect(task, view, inv)
        elif task['kind'] == 'delivery':
            target, interact, stage = self._plate_and_collect(task, view, inv)
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
