import gymnasium as gym
import vizdoom as vzd
import numpy as np
from gymnasium import spaces

class VizDoomDeathmatchEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.game = vzd.DoomGame()
        self.game.load_config(vzd.scenarios_path + "/deathmatch.cfg")
        self.game.set_window_visible(True)
        self.game.init()

        # Anzahl Actions aus VizDoom
        n = self.game.get_available_buttons_size()

        # Action Space (binary buttons)
        self.action_space = spaces.MultiBinary(n)

        # Observation Space (Bild)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(480, 640, 3),
            dtype=np.uint8
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.game.new_episode()
        state = self.game.get_state()

        if state is None:
            return np.zeros((480, 640, 3), dtype=np.uint8), {}

        return state.screen_buffer, {}

    def step(self, action):
        reward = self.game.make_action(action.tolist())

        done = self.game.is_episode_finished()

        if not done:
            state = self.game.get_state()
            obs = state.screen_buffer
        else:
            obs = np.zeros((480, 640, 3), dtype=np.uint8)

        return obs, reward, done, False, {}

    def render(self):
        pass  # VizDoom rendert selbst

    def close(self):
        self.game.close()