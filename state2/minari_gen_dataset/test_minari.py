import gymnasium as gym
import numpy as np

from stable_baselines3 import SAC

import minari
from minari import DataCollector

from tqdm.auto import tqdm
env = gym.make("Pendulum-v1")
env = DataCollector(env)

model = SAC("MlpPolicy", env, verbose=1,tensorboard_log="./tensorboard/")
model = SAC.load("sac_pendulum")


total_episodes = 1_000
for i in tqdm(range(total_episodes)):
    obs, _ = env.reset(seed=42)
    while True:
        action, _  = model.predict(obs)
        obs, rew, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

dataset = env.create_dataset(
    dataset_id="pendulum/sac-v0",
    algorithm_name="ExpertPolicy",
    author="sunwuzhou03",
)
