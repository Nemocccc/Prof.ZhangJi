import gymnasium as gym
import numpy as np

from stable_baselines3 import SAC

env = gym.make("Pendulum-v1")

model = SAC("MlpPolicy", env, verbose=1,tensorboard_log="./tensorboard/")
model.learn(total_timesteps=10000, log_interval=4)
model.save("sac_pendulum")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_pendulum")

obs,info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated,truncated, info = env.step(action)
    env.render()
    done=terminated or truncated
    if done:
      obs = env.reset()
