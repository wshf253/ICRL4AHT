"""Diagnostic rollout of upstream heuristics (privileged full-state access).

Run from the repository root with python -m scripts.preview_teammates.
This is not an ICRL evaluation or an observation-only agent baseline.
"""
import argparse
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw

from agents.overcooked_v2 import (
    AssemblyLineAgentV2, AssemblyLineTheta, TerritoryAgentV2,
    UtilityGreedyAgentV2, RecipeAwareButtonAgentV2,
)
from envs import make_env
from envs.overcooked_v2.common import Actions


FAMILIES = ('assembly_line', 'territory', 'utility_greedy', 'recipe_aware_button')


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def build_policy(family, layout, role):
    if family == 'assembly_line':
        theta = {
            'flex': AssemblyLineTheta.default,
            'runner': AssemblyLineTheta.ingredient_runner,
            'plater': AssemblyLineTheta.plater_deliverer,
        }[role]()
        return AssemblyLineAgentV2(layout, theta=theta)
    return {
        'territory': TerritoryAgentV2,
        'utility_greedy': UtilityGreedyAgentV2,
        'recipe_aware_button': RecipeAwareButtonAgentV2,
    }[family](layout)


def contents(value):
    value = int(value)
    ingredients, packed, idx = [], value >> 2, 0
    while packed:
        ingredients += [str(idx)] * (packed & 3)
        packed >>= 2
        idx += 1
    return ('P' if value & 1 else '') + ('C' if value & 2 else '') + ','.join(ingredients) or '-'


def frame(state, layout, step, score):
    """Full-state diagnostic view; never use these frames as partial observations."""
    grid = np.asarray(state.grid)
    h, w = grid.shape[:2]
    tile, top = 64, 64
    image = Image.new('RGB', (max(w * tile, 640), top + h * tile + 90), '#f7f7f7')
    draw = ImageDraw.Draw(image)
    draw.text((8, 6), f'{layout} | step {step} | shared return {score:g}', fill='black')
    draw.text((8, 25), 'DEBUG: full state; both upstream heuristics have privileged access', fill='black')
    labels = {0: '', 1: '#', 4: 'SERVE', 5: 'POT', 6: 'RECIPE', 7: 'BUTTON', 9: 'PLATES'}
    for y in range(h):
        for x in range(w):
            obj, dyn, extra = map(int, grid[y, x])
            box = (x * tile, top + y * tile, (x+1) * tile-1, top+(y+1)*tile-1)
            draw.rectangle(box, fill='#dddddd' if obj == 1 else '#ffffff', outline='#888888')
            label = f'ING {obj-10}' if obj >= 10 else labels.get(obj, str(obj))
            draw.text((box[0]+3, box[1]+3), label, fill='black')
            if dyn:
                draw.text((box[0]+3, box[1]+24), contents(dyn), fill='#196019')
            if extra:
                draw.text((box[0]+3, box[1]+42), f't:{extra}', fill='#555555')
    # Direction enum: up, down, right, left.
    directions = {0: (0, -16), 1: (0, 16), 2: (16, 0), 3: (-16, 0)}
    for i, (x, y) in enumerate(zip(np.asarray(state.agents.pos.x), np.asarray(state.agents.pos.y))):
        cx, cy = int(x)*tile+32, top+int(y)*tile+32
        draw.ellipse((cx-17, cy-17, cx+17, cy+17), fill=['#c84040', '#367ac4'][i % 2])
        dx, dy = directions[int(state.agents.dir[i])]
        draw.line((cx, cy, cx+dx, cy+dy), fill='white', width=3)
        draw.text((cx-4, cy-5), str(i), fill='white')
    bottom = top+h*tile
    inventory = [contents(v) for v in np.asarray(state.agents.inventory)]
    draw.text((8, bottom+8), f'Inventory: {inventory} | recipe: {contents(state.recipe)}', fill='black')
    draw.text((8, bottom+28), 'P=plate; C=cooked; digits=ingredient IDs; t=timer/indicator', fill='black')
    draw.text((8, bottom+48), '0=red agent; 1=blue agent; white line=facing direction', fill='black')
    return np.asarray(image)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layout', default='grounded_coord_ring')
    p.add_argument('--steps', type=positive, default=400)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--agent0', choices=FAMILIES, default='utility_greedy')
    p.add_argument('--agent1', choices=FAMILIES, default='territory')
    p.add_argument('--role0', choices=('flex', 'runner', 'plater'), default='flex')
    p.add_argument('--role1', choices=('flex', 'runner', 'plater'), default='flex')
    p.add_argument('--video', choices=('mp4', 'none'), default='mp4')
    p.add_argument('--fps', type=positive, default=8)
    p.add_argument('--out', type=Path, default=Path('outputs/preview'))
    args = p.parse_args()
    for family, role in ((args.agent0, args.role0), (args.agent1, args.role1)):
        if family != 'assembly_line' and role != 'flex':
            p.error('role selection is only supported for assembly_line')
    args.out.mkdir(parents=True, exist_ok=True)
    # Avoid silently mixing logs from different runs.
    for name in ('rollout.jsonl', 'summary.json', 'rollout.mp4'):
        if (args.out/name).exists():
            p.error(f'{args.out/name} exists; choose a fresh --out directory')
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    dirty = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
    env = make_env('overcooked-v2', {'layout': args.layout, 'max_steps': args.steps})
    policies = [build_policy(args.agent0, env.layout, args.role0),
                build_policy(args.agent1, env.layout, args.role1)]
    key = jax.random.PRNGKey(args.seed)
    key, reset_key, *policy_keys = jax.random.split(key, 4)
    obs, wrapped = env.reset(reset_key)
    state = wrapped.env_state
    hstates = [policy.init_agent_state(i).replace(rng_key=policy_keys[i])
               for i, policy in enumerate(policies)]
    # Use the raw transition with factory settings to retain the terminal frame.
    # The benchmark wrapper auto-resets at termination.
    transition = jax.jit(env.env.step_env)
    returns = {a: 0.0 for a in env.agents}
    shaped = {a: 0.0 for a in env.agents}
    delivery_steps = 0
    terminated = False
    writer = None
    if args.video == 'mp4':
        import imageio.v2 as imageio
        writer = imageio.get_writer(args.out/'rollout.mp4', fps=args.fps,
                                   codec='libx264', macro_block_size=2)
    try:
        if writer is not None:
            writer.append_data(frame(state, args.layout, 0, 0))
        with (args.out/'rollout.jsonl').open('w') as log:
            for step in range(args.steps):
                actions = {}
                for i, name in enumerate(env.agents):
                    action, hstates[i] = policies[i].get_action(obs[name], state, hstates[i])
                    actions[name] = jnp.asarray(action, dtype=jnp.int32)
                key, step_key = jax.random.split(key)
                obs, state, rewards, dones, info = transition(step_key, state, actions)
                correct = bool(state.new_correct_delivery)
                delivery_steps += int(correct)
                for name in env.agents:
                    returns[name] += float(rewards[name])
                    shaped[name] += float(info['shaped_reward'][name])
                terminated = bool(dones['__all__'])
                record = {
                    'step': step+1, 'actions': {a: int(v) for a, v in actions.items()},
                    'action_names': {a: Actions(int(v)).name for a, v in actions.items()},
                    'rewards': {a: float(v) for a, v in rewards.items()},
                    'correct_delivery_this_step': correct, 'done': terminated,
                    'positions': [[int(state.agents.pos.x[i]), int(state.agents.pos.y[i])]
                                  for i in range(env.num_agents)],
                    'inventories': np.asarray(state.agents.inventory).tolist(),
                }
                log.write(json.dumps(record)+'\n')
                if writer is not None:
                    writer.append_data(frame(state, args.layout, step+1, returns[env.agents[0]]))
                if terminated:
                    break
    finally:
        if writer is not None:
            writer.close()
    if not terminated:
        raise RuntimeError('Configured episode horizon did not terminate')
    summary = {
        'status': 'completed', 'config': {**vars(args), 'out': str(args.out)},
        'git_commit': revision.stdout.strip() or None,
        'git_dirty': bool(dirty.stdout.strip()),
        'jax_version': jax.__version__, 'devices': [str(d) for d in jax.devices()],
        'steps': step+1, 'returns_per_agent': returns,
        'shared_return': returns[env.agents[0]],
        'shaped_returns_per_agent': shaped,
        'steps_with_correct_delivery': delivery_steps,
        'agent_view_size': env.env.agent_view_size,
        'policy_information': 'privileged_full_state_for_both_heuristics',
        'note': 'Diagnostic only; not ICRL evaluation. Shared rewards are not summed across agents. '
                'Delivery metric counts timesteps, not dishes (simultaneous deliveries may occur).',
    }
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
