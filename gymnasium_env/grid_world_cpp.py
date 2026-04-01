from typing import Optional
import numpy as np
import gymnasium as gym

import pygame

#
# Coverage Path Planning (CPP) environment for a 2D grid world with obstacles.
#
# Inspired by:
#   - Santos et al. (2023). "A Deep Reinforcement Learning Approach for the Patrolling
#     Problem of Water Resources Through Autonomous Surface Vehicles: The Ypacarai Lake Case".
#   - Kiran et al. (2021). "A Comprehensive Survey on Coverage Path Planning for
#     Mobile Robots in Dynamic Environments".
#
# Unlike the navigation GridWorld (grid_world_obstacles.py), there is no target cell.
# The agent's goal is to visit every free (non-obstacle) cell at least once.
#
# ----- Original reward function (grid_world_obstacles.py) -----
# The original environment rewards the agent for reaching a single target cell:
#   +10.0  when the agent reaches the target
#   -10.0  when the episode is truncated (max_steps exceeded)
#    else   prev_distance - current_distance - 0.1   (distance shaping + step cost)
#
# This sparse, distance-based signal is appropriate for point-to-point navigation but
# gives no incentive to systematically explore the entire grid, making it unsuitable
# for CPP tasks.
#
# ----- New CPP reward function (this file) -----
# Reward components are designed around *coverage gain* rather than goal proximity:
#
#   +1.0   for each previously unvisited free cell the agent enters
#           (exploration reward — encourages visiting new areas)
#   -0.02  step cost applied every step
#           (efficiency incentive — discourages unnecessary wandering)
#   +10.0  one-time completion bonus when 100 % of free cells have been visited
#           (goal signal — makes full coverage the clear objective)
#   -5.0   truncation penalty if the episode ends before full coverage
#           (motivates finishing within the step budget)
#
# Rationale (grounded in the referenced literature):
#   - The per-cell exploration reward mirrors the "information gain" reward used in
#     patrolling and coverage literature, where the agent is rewarded proportionally
#     to the new area it uncovers (Santos et al., 2023).
#   - The step cost + completion bonus combination follows the standard CPP formulation
#     described in the survey (Kiran et al., 2021), where the agent must balance
#     thoroughness (cover everything) against efficiency (minimize path length).
#   - Revisiting cells yields only the step cost (no extra penalty), avoiding overly
#     punishing the agent when backtracking is geometrically unavoidable (e.g., after
#     exploring a corridor with a single entrance).
#
# ----- Observation space -----
# The observation is a flat array of length 2 + size*size:
#   obs[0:2]      — agent's (x, y) location
#   obs[2:]       — flattened coverage map (row-major), where:
#                     0 = free, not yet visited
#                     1 = free, already visited
#                     2 = obstacle
#
# This gives the agent full knowledge of its own position and of which cells still
# need to be covered, enabling it to plan systematic coverage paths.


class GridWorldCPPEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size: int = 5, obs_quantity: int = 3, max_steps: int = 200):
        self.size = size
        self.window_size = 512
        self.obs_quantity = obs_quantity
        self.obstacles_locations = []
        self.count_steps = 0
        self.max_steps = max_steps

        self._agent_location = np.array([-1, -1], dtype=int)

        # Coverage map: 0 = unvisited free, 1 = visited, 2 = obstacle
        self._coverage_map = np.zeros((size, size), dtype=int)
        self._total_free_cells = 0
        self._visited_cells = 0

        # Observation: agent location (2) + flattened coverage map (size * size)
        self.observation_space = gym.spaces.Box(
            low=0, high=2, shape=(2 + size * size,), dtype=int
        )

        # 4 actions: right, up, left, down
        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),   # right
            1: np.array([0, -1]),  # up
            2: np.array([-1, 0]), # left
            3: np.array([0, 1]),   # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        obs = list(self._agent_location) + list(self._coverage_map.flatten())
        return np.array(obs, dtype=int)

    def _get_info(self):
        coverage_ratio = (
            self._visited_cells / self._total_free_cells
            if self._total_free_cells > 0
            else 0.0
        )
        return {
            "coverage_ratio": coverage_ratio,
            "visited_cells": self._visited_cells,
            "total_free_cells": self._total_free_cells,
            "size": self.size,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.count_steps = 0
        self.obstacles_locations = []
        self._coverage_map = np.zeros((self.size, self.size), dtype=int)

        # Place agent at a random free cell
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Place obstacles (not on top of the agent)
        for _ in range(self.obs_quantity):
            obstacle_location = self._agent_location.copy()
            while np.array_equal(obstacle_location, self._agent_location) or any(
                np.array_equal(obstacle_location, loc) for loc in self.obstacles_locations
            ):
                obstacle_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            self.obstacles_locations.append(obstacle_location)
            self._coverage_map[obstacle_location[0], obstacle_location[1]] = 2

        # Mark starting cell as visited
        self._coverage_map[self._agent_location[0], self._agent_location[1]] = 1
        self._total_free_cells = int(np.sum(self._coverage_map != 2))
        self._visited_cells = 1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        old_location = self._agent_location.copy()

        # Move agent, clipping to grid bounds
        new_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        # Revert movement if the destination is an obstacle
        if any(np.array_equal(new_location, loc) for loc in self.obstacles_locations):
            new_location = old_location

        self._agent_location = new_location
        self.count_steps += 1

        cell_x, cell_y = self._agent_location[0], self._agent_location[1]

        # --- CPP reward function ---
        if self._coverage_map[cell_x, cell_y] == 0:
            # New free cell: grant exploration reward and mark as visited
            self._coverage_map[cell_x, cell_y] = 1
            self._visited_cells += 1
            reward = 1.0
        else:
            # Revisiting a cell: only the step cost applies
            reward = 0.0

        # Step cost applied every step
        reward -= 0.02

        # Check whether all free cells have been covered
        terminated = self._visited_cells == self._total_free_cells

        if terminated:
            reward += 10.0  # Completion bonus

        if self.count_steps >= self.max_steps and not terminated:
            truncated = True
            reward -= 5.0  # Truncation penalty
        else:
            truncated = False

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Coverage Path Planning")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        # Draw cells according to the coverage map
        for x in range(self.size):
            for y in range(self.size):
                cell_val = self._coverage_map[x, y]
                if cell_val == 2:
                    color = (0, 0, 0)           # Obstacle: black
                elif cell_val == 1:
                    color = (144, 238, 144)     # Visited: light green
                else:
                    color = (255, 255, 255)     # Unvisited: white

                pygame.draw.rect(
                    canvas,
                    color,
                    pygame.Rect(
                        pix_square_size * x,
                        pix_square_size * y,
                        pix_square_size,
                        pix_square_size,
                    ),
                )

        # Draw agent as a blue circle
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (128, 128, 128),
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                (128, 128, 128),
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
