from enviroment.vizdoom_env import VizDoomDeathmatchEnv
import time

env = VizDoomDeathmatchEnv()

obs, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # zufällige Aktion
    obs, reward, done, truncated, info = env.step(action)
    
    # Frame sichtbar halten
    time.sleep(0.03)  # ca. 30 FPS

env.close()