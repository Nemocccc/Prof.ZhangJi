import gymnasium as gym
import numpy as np

from stable_baselines3 import SAC

env = gym.make("Pendulum-v1",render_mode="human")

model = SAC("MlpPolicy", env, verbose=1,tensorboard_log="./tensorboard/")

model = SAC.load("sac_pendulum")

obs,info = env.reset()
while True:
    # print(obs)
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated,truncated, info = env.step(action)
    env.render()
    done=terminated or truncated
    if done:
      break