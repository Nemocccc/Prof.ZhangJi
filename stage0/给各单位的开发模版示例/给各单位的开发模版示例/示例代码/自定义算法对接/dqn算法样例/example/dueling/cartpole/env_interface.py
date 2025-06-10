from typing import Any, Dict, List, Union, Tuple, DefaultDict

import gym
from gym import spaces

from drill import summary
from drill.env import Env
from drill.pipeline.interface import ObsData
from drill.keys import REWARD
from collections import defaultdict

ENV_NAME = 'CartPole-v1'


class AtariEnv(Env):
    def __init__(self, env_id: int, extra_info: dict):
        self._env = gym.make(ENV_NAME)
        # self._env = gym.make(ENV_NAME, render_mode='human')
        self._global_raw_reward_scalar = None
        self._episode_reward_dict = defaultdict(float)
        self._MAX_STEP = 1000
        self._step_num = 0

    @property
    def agent_names(self) -> List[str]:
        return ['atari']

    @property
    def action_space(self) -> spaces.Discrete:
        return self._env.action_space

    @property
    def observation_space(self) -> spaces.Box:
        return self._env.observation_space

    def reset(self) -> Any:
        self._episode_reward_dict = defaultdict(float)
        self._step_num = 0
        obs = self._env.reset()
        return {self.agent_names[0]: ObsData(obs, {})}

    def step(self, action_dict: Dict[str, Any]) -> Tuple[Dict[str, ObsData], bool]:
        agent_name = list(action_dict.keys())[0]
        action_name = list(action_dict[agent_name].keys())[0]
        command = action_dict[agent_name][action_name]
        if gym.__version__ > '0.26.0':
            obs, reward, done, _, info = self._env.step(command)
        else:
            obs, reward, done, info = self._env.step(command)

        self._step_num += 1
        self._episode_reward_dict[agent_name + 'raw_reward'] += reward
        if self._step_num > self._MAX_STEP or done:
            done = True
            summary.average("episode_reward", self._episode_reward_dict[agent_name + "raw_reward"])

        return {self.agent_names[0]: ObsData(obs, {REWARD: reward})}, done
