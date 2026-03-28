import gymnasium as gym
import vizdoom as vzd
import numpy as np
from gymnasium import spaces

#https://gymnasium.farama.org/index.html 
#https://gymnasium.farama.org/introduction/create_custom_env/
#Minimalste env für viz Deathmatch

class VizDoomDeathmatchEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.game = vzd.DoomGame()
        self.game.load_config(vzd.scenarios_path + "/deathmatch.cfg")
        self.game.set_window_visible(True)
        self.game.init()

        #missing actions, obs, step, render, reset
        #ohne die kann weder observiert noch gestept (simuliert) werden