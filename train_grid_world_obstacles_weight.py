#
# python train_grid_world_obstacles.py <train|test|run>
#

import gymnasium as gym
from gymnasium_env.grid_world_obstacles_weight import GridWorldRenderEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from datetime import datetime
import sys

def print_action(action: int) -> str:
    return {
        0: "right",
        1: "up",
        2: "left",
        3: "down",
    }.get(action, "unknown")

if len(sys.argv) < 2 or sys.argv[1] not in ['train', 'test', 'run']:
    print("Usage: python train_grid_world_obstacles.py <train|test|run>")
    sys.exit(1)

mode = sys.argv[1]

try:
    gym.register(
        id="gymnasium_env/GridWorld-v1",
        entry_point=GridWorldRenderEnv,
    )
except Exception:
    pass

# --- Hyperparameters ---
DIM = 20
OBSTACLES = 40
MAX_STEPS = 500
TOTAL_TIMESTEPS = 500_000
ENTROPY_COEF = 0.02
# -----------------------

if mode == 'train':
    print("--- Starting Training ---")
    env = gym.make(
        "gymnasium_env/GridWorld-v1",
        size=DIM,
        obs_quantity=OBSTACLES,
        max_steps=MAX_STEPS,
        render_mode="rgb_array"
    )
    check_env(env)

    model = PPO("MlpPolicy", env, verbose=1, ent_coef=ENTROPY_COEF, device="cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'log/ppo_obstacles_{DIM}_{OBSTACLES}_{MAX_STEPS}_{ENTROPY_COEF}_{timestamp}'
    model_path = f'data/ppo_obstacles_{DIM}_{OBSTACLES}_{MAX_STEPS}_{ENTROPY_COEF}_{timestamp}.zip'

    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    print(f"Starting learning with {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(model_path)
    print(f"Model trained and saved to {model_path}")
    print(f"Logs saved to {log_dir}")

elif mode == 'run':
    model_name = input("Enter model filename (e.g., ppo_obstacles_20_40_500_0.02_20250924_103000): ")
    model_path = f'data/{model_name}.zip'
    print(f'--- Loading model from {model_path} for a run ---')

    model = PPO.load(model_path)
    env = gym.make(
        "gymnasium_env/GridWorld-v1",
        size=DIM,
        obs_quantity=OBSTACLES,
        max_steps=MAX_STEPS,
        render_mode="human"
    )

    (obs, _) = env.reset()
    done = False
    truncated = False
    steps = 0
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action.item())
        print(f"Step: {steps+1}, Action: {print_action(action.item())}, Reward: {reward:.2f}, Done: {done}, Truncated: {truncated}")
        steps += 1
    print("--- Run Finished ---")

elif mode == 'test':
    model_name = input("Enter model filename (e.g., ppo_obstacles_20_40_500_0.02_20250924_103000): ")
    model_path = f'data/{model_name}.zip'
    print(f'--- Loading model from {model_path} for testing ---')

    model = PPO.load(model_path)
    env = gym.make(
        "gymnasium_env/GridWorld-v1",
        size=DIM,
        obs_quantity=OBSTACLES,
        max_steps=MAX_STEPS,
        render_mode="rgb_array" # No rendering for faster testing
    )

    num_episodes = 100
    success_count = 0
    for i in range(num_episodes):
        (obs, _) = env.reset()
        done = False
        truncated = False
        steps = 0
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action.item())
            steps += 1
        
        if done and not truncated: # Reached the goal
            success_count += 1
            print(f"Episode {i+1}: Success in {steps} steps.")
        else:
            print(f"Episode {i+1}: Failed to reach goal.")

    success_rate = (success_count / num_episodes) * 100
    print(f"--- Test Finished ---")
    print(f"Success Rate: {success_rate:.2f}% ({success_count}/{num_episodes})")