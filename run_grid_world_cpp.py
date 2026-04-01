"""
Random-agent test for the Coverage Path Planning environment.

Runs a single episode on a small 5x5 grid so the coverage progress is easy
to observe.  No training is involved — the agent picks actions uniformly at
random.  The script prints per-step diagnostics and a final summary.

Usage:
    python run_grid_world_cpp.py           # headless (no window)
    python run_grid_world_cpp.py render    # with pygame window
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gymnasium_env.grid_world_cpp import GridWorldCPPEnv

# ── configuration ────────────────────────────────────────────────────────────
GRID_SIZE  = 5
OBSTACLES  = 3
MAX_STEPS  = 200
SEED       = 42
RENDER     = "human" if len(sys.argv) > 1 and sys.argv[1] == "render" else None
# ─────────────────────────────────────────────────────────────────────────────

env = GridWorldCPPEnv(
    render_mode=RENDER,
    size=GRID_SIZE,
    obs_quantity=OBSTACLES,
    max_steps=MAX_STEPS,
)

observation, info = env.reset(seed=SEED)

print("=" * 55)
print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}  |  Obstacles: {OBSTACLES}  |  Max steps: {MAX_STEPS}")
print(f"  Free cells to cover: {info['total_free_cells']}")
print(f"  Agent starts at:     {observation[:2]}")
print("=" * 55)

total_reward = 0.0

for step in range(MAX_STEPS):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    cov = info["coverage_ratio"]
    vis = info["visited_cells"]
    tot = info["total_free_cells"]
    print(
        f"step {step + 1:>3}  action={action}  reward={reward:+.3f}  "
        f"coverage={cov:.0%} ({vis}/{tot})"
    )

    if terminated or truncated:
        break

print("=" * 55)
if terminated:
    print("  Result : ALL CELLS COVERED")
else:
    print(f"  Result : truncated — coverage {info['coverage_ratio']:.0%}")
print(f"  Total reward : {total_reward:.3f}")
print("=" * 55)

env.close()
