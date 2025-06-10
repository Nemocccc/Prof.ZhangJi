from drill.env import Env
from drill import summary
from raw_env.env_def import *
from raw_env.f4v1_game import make_env_f4v1
from typing import Any, DefaultDict, Dict, List, Tuple
from drill.pipeline.interface import ObsData


class F4v1Env(Env):

    def __init__(self, env_id: int, extra_info: dict):
        self.env_id = env_id
        self._env = make_env_f4v1()
        self.extra_info = extra_info
        from configs.builder_config import AGENT_NAMES
        self.agent_names = AGENT_NAMES
        # self.reset()
        print("Environment is ready!")

    def reset(self) -> Any:
        """重置 Env

        Returns
        -------
        Any
            环境的初始状态
        """

        obs = self._reset_env()
        obs_dict = {agent_name: ObsData(obs=obs[agent_name]) for agent_name in self.agent_names}
        return obs_dict

    def step(self, command_dict: Dict[str, Any]) -> Tuple[Dict[str, ObsData], bool]:
        """所有参与者依次执行 command

        Parameters
        ----------
        command_dict : Dict[str, Any]
            包含所有参与者的动作

        Returns
        -------
        Tuple[Dict[str, Any], bool, Dict[str, Any]]
            obs_dict, done, reward_dict
        """

        # 向仿真端发送命令
        # try:
        actions = []
        from configs.builder_config import AGENT_NAMES
        for agent_name in AGENT_NAMES:
            actions.append([command_dict.get(agent_name, None)])
        self.raw_obs, self._reward, self.done, self.ob_info = self._env.step(actions, VIS_STEPS)
        obs_dict = {agent_name: ObsData(self.raw_obs[agent_name],
                                        {"reward": self._reward[int(agent_name[-1])],},
                                        agent_name)
                    for agent_name in self.agent_names}

        if self.done:
            for agent_name in self.agent_names:
                # 默认以时间作为横轴记录数据
                summary.average(f"{agent_name}_reward_" + self.extra_info['index'], self._reward[int(agent_name[-1])])
            summary.average("total_step_" + self.extra_info['index'], self.ob_info['eplen'])
            summary.average("final_v_" + self.extra_info['index'], sum(self.ob_info['final_v']))
            summary.average("hit_rate_" + self.extra_info['index'], sum(self.ob_info['pldone']))
            summary.average(name="episode_reward(time)_" + self.extra_info['index'], value=self.ob_info['eprew'])
            # 如果需要以learn_step作为横轴记录数据，则需要传递-指定模型学习步数-作为横轴参数
            summary.average(name="episode_reward(learn_step)_" + self.extra_info['index'], value=self.ob_info['eprew'],
                            x_axis="f4v1_model_learn_step")
            print(f'Env {self.env_id}: The total number of steps in the current episode: {self._env.step_cum}')

        return obs_dict, self.done

    def _reset_env(self):
        # 重置为初始状态
        self.raw_obs = self._env.reset()
        self.done = False
        self.error = False
        self._reward = [0, 0, 0, 0]
        self.ob_info = {'eprew': 0, 'eplen': 0, 'pldone': [False, False, False, False], 'final_v': [0, 0, 0, 0]}
        return self.raw_obs
