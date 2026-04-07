from typing import Optional
import numpy as np
import gymnasium as gym

import pygame

#
# This code is based on the example from Gymnasium: 
# https://gymnasium.farama.org/introduction/create_custom_env/
#
# This environment implements a simple grid world with obstacles.
# The agent (blue circle) must reach the target (red square) in as few steps as possible.
#
# The state is represented as a flattened array containing:
# - the agent's (x, y) location
# - the target's (x, y) location
# - the state of the 4 neighboring cells (up, down, left, right),
#   where 0 indicates a free cell and 1 indicates an obstacle or wall.
#
# The action space is discrete with 4 actions: move right, up, left, down.
#
# Reward scheme (non-homogeneous, Interest Gradient):
# - Compute R = W * R_abs (element-wise), where:
#   - R_abs: static per episode (goal=100, free=20, obstacle=0)
#   - W: temporal mask in [0,1] (free/goal start at 1, obstacles at 0)
# - Legal move reward is a linear mapping of grad_R = R[next] - R[curr]
# - Illegal action (wall/obstacle): reward = -C_illegal and the agent does not move
# - Episode terminates when the agent reaches the goal (or truncates at max_steps)

class GridWorldRenderEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size: int = 5, obs_quantity: int = 5, max_steps: int = 100):
        # The size of the square grid
        self.size = size
        self.window_size = 512
        self.obs_quantity = obs_quantity
        self.obstacles_locations = []
        self.count_steps = 0
        self.max_steps = max_steps

        # --- Reward (Interest Gradient + Temporal Mask) ---
        # Temporal mask regeneration rate
        self.delta = 0.055
        # Reward bounds for legal moves
        self.r_max = 5.0
        self.r_min = -0.5
        # Penalty for illegal action (wall/obstacle)
        self.C_illegal = 10.0
        # Gradient bounds (can be negative)
        self.R_max = 100.0
        self.R_min = -100.0

        # Absolute importance values (R_abs)
        self.R_abs_goal = 100.0
        self.R_abs_free = 20.0

        # Matrices (initialized in reset)
        self.R_abs = None
        self.W = None
        self._obstacle_mask = None
        self._non_obstacle_mask = None

        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self._agent_location = np.array([-1, -1], dtype=int)
        self._target_location = np.array([-1, -1], dtype=int)
        self._neighbors = np.array([0,0,0,0], dtype=int)  # right, up, left, down

        # The state is represented with the agent's and target's location and the grid of neighbors
        self.observation_space = gym.spaces.Box(0, size - 1, shape=(2 + 2 + 4,), dtype=int)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(4)
        # Dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, -1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, 1]),  # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        flattened = []
        flattened.extend(self._agent_location)
        flattened.extend(self._target_location)
        flattened.extend(self._neighbors)
        return np.array(flattened, dtype=int)

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            ),
            "size": self.size
        }

    def set_neighbors(self, obstacles_locations):
        # create a map of the neighbors
        # 0 = free, 1 = obstacle or wall
        directions = [np.array([1, 0]), np.array([0, -1]), np.array([-1, 0]), np.array([0, 1])]
        for i, direction in enumerate(directions):
            neighbor = self._agent_location + direction
            if (0 <= neighbor[0] < self.size) and (0 <= neighbor[1] < self.size) and not any(np.array_equal(neighbor, loc) for loc in obstacles_locations):
                self._neighbors[i] = 0
            else:
                self._neighbors[i] = 1

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.count_steps = 0
        self.obstacles_locations = []

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )        

        for _ in range(self.obs_quantity):
            obstacle_location = self._agent_location
            while (np.array_equal(obstacle_location, self._agent_location) or 
                   np.array_equal(obstacle_location, self._target_location) or
                   any(np.array_equal(obstacle_location, loc) for loc in self.obstacles_locations)):
                obstacle_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            self.obstacles_locations.append(obstacle_location)

        # --- Initialize Interest Matrices (R_abs and W) ---
        self._obstacle_mask = np.zeros((self.size, self.size), dtype=bool)
        for obs in self.obstacles_locations:
            x_obs, y_obs = int(obs[0]), int(obs[1])
            self._obstacle_mask[x_obs, y_obs] = True
        self._non_obstacle_mask = ~self._obstacle_mask

        # Absolute importance (static per episode)
        self.R_abs = np.full((self.size, self.size), fill_value=self.R_abs_free, dtype=np.float32)
        self.R_abs[self._obstacle_mask] = 0.0
        x_goal, y_goal = int(self._target_location[0]), int(self._target_location[1])
        self.R_abs[x_goal, y_goal] = self.R_abs_goal

        # Temporal mask (short-term memory)
        self.W = np.ones((self.size, self.size), dtype=np.float32)
        self.W[self._obstacle_mask] = 0.0

        self.set_neighbors(self.obstacles_locations)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def distance(self, location, target):
        x = (location[0] - target[0])*(location[0] - target[0])
        y = (location[1] - target[1])*(location[1] - target[1])
        return np.sqrt(x+y)

    def step(self, action):

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]

        current_location = self._agent_location.copy()
        proposed_location = current_location + direction

        x_next, y_next = int(proposed_location[0]), int(proposed_location[1])
        out_of_bounds = not (0 <= x_next < self.size and 0 <= y_next < self.size)
        hits_obstacle = False
        if not out_of_bounds:
            hits_obstacle = bool(self._obstacle_mask[x_next, y_next])

        illegal_action = out_of_bounds or hits_obstacle
        terminated = False

        if illegal_action:
            # Illegal move: no state change, fixed penalty
            reward = -float(self.C_illegal)
        else:
            # Compute current relative importance R = W * R_abs (element-wise)
            R = self.W * self.R_abs
            x_curr, y_curr = int(current_location[0]), int(current_location[1])

            # Interest gradient between current cell and next cell
            grad_R = float(R[x_next, y_next] - R[x_curr, y_curr])

            # Map grad_R from [R_min, R_max] to [r_min, r_max]
            reward = ((self.r_max - self.r_min) / (self.R_max - self.R_min)) * (grad_R - self.R_min) + self.r_min

            # Apply the movement
            self._agent_location = proposed_location
            terminated = np.array_equal(self._agent_location, self._target_location)

            # Update temporal mask W for next time step (t+1)
            self.W[self._non_obstacle_mask] = np.minimum(1.0, self.W[self._non_obstacle_mask] + self.delta)
            self.W[x_next, y_next] = 0.0

        self.set_neighbors(self.obstacles_locations)

        self.count_steps += 1

        truncated = self.count_steps >= self.max_steps and not terminated

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, float(reward), terminated, truncated, info
    
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Draw the obstacles
        for obs in self.obstacles_locations:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * obs,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )    
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()