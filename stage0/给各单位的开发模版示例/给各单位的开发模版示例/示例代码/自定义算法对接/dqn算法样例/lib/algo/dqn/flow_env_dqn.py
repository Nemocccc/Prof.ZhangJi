from __future__ import annotations

from typing import Dict, List

import numpy as np

from drill.keys import REWARD, DONE, ACTION, ADVANTAGE
from lib.core.flow_env.vb import FlowEnvVB
from lib.core.keys import OBS, OBS_NEXT


class FlowEnvDQN(FlowEnvVB):

    def __init__(self, environment_description):
        super().__init__(environment_description)

    def enhance_fragment(self, agent_name: str, fragments: List[Dict]):
        """ 对不断 `observe` 和 `step` 收集的数据进行处理

        什么时候调用这个方法？
        一次 `observe` 和 `step` 收集的数据记为一个 fragment，当收集到的数据达到
        `fragment_size` （一个配置参数）时调用此方法

        注意: fragments 只能原地修改，这个方法不接受返回值，这是由 flow 决定的

        Parameters
        ----------
        agent_name : str
            agent 的名字
        fragments : List[Dict]
            长度为 `fragment_size`，每一个元素都是 3 元组，分别对应 `observe` 的
            返回值（准确的说是 `observe_return[agent_name]["obs"]`，`step` 的参数
            `predict_output` 和 `step` 的返回值。
        """

        # 对于多智能体竞争环境存在一种情况：
        # 采集 fragment 的顺序：先依次执行FlowEnv.step() 和 FlowEnv.observe() 得
        # 到 state, reward, 然后执行 FlowModel.predict() 得到 action, value，然后采集
        # 一个 fragment(state, reward, action, value)，然后执行下一个 FlowEnv.step()。
        # 设 fragment_size=64+1 = 65，第 64 次执行 FlowEnv.step() 时，已
        # 经采集了 64 个fragment（若某个 agent 在第 64 次 step done了，返回的
        # obs, reward, done 不为空），开始采集第 65 个fragment，然后会执行第 65 个
        # FlowEnv.step(), 再执行 FlowEnv.enhance_fragment()，此时进入 enhance_fragment()
        # 后会遇到问题: 此时的 self._env_info 已经不包含刚刚 done 的 agent 了，因此 last_reward
        # = None。因此额外使用 self._final_reward 维护所有 done 的 agent 的最后一个 reward。

        model_name = self._builder.get_model_name(agent_name)
        n_step = self._get_n_step(model_name)
        discount = self._get_gamma(model_name)

        obs, act, rew, d, fragments = self.get_info_to_list(fragments)

        target_obs, actions, rewards, dones, invalid_idx = self.cal_n_step_value(obs, act, rew, d,
                                                                                 discount, n_step)
        invalid_idx.reverse()
        for idx in invalid_idx:
            fragments.pop(idx)

        reward_len = len(rewards)
        fragments_len = len(fragments)

        assert reward_len == fragments_len

        for i in range(reward_len):
            fragments[i][0][REWARD] = np.asarray(rewards[i], dtype=np.float32)
            fragments[i][0][DONE] = np.asarray(dones[i], dtype=np.float32)
            fragments[i][0].update({OBS_NEXT: target_obs[i]})
            fragments[i].append({ADVANTAGE: np.asarray(1., dtype=np.float32)})  # 仅用于占位

    def get_info_to_list(self, fragments):

        def process_flow_state(old_state):
            new_state = {OBS: {}}
            for k, v in old_state.items():
                if k == REWARD:
                    new_state[REWARD] = v
                elif k == DONE:
                    new_state[DONE] = v
                else:
                    new_state[OBS][k] = v
            return new_state

        obs = []
        actions = []
        rewards = []
        dones = []

        for i in range(len(fragments)):
            flow_state, flow_action_dict, _ = fragments[i]
            new_flow_state = process_flow_state(flow_state)
            obs.append(new_flow_state[OBS])
            actions.append(flow_action_dict[ACTION])
            rewards.append(new_flow_state[REWARD])
            dones.append(new_flow_state[DONE])
            fragments[i][0] = new_flow_state
        return obs, actions, rewards, dones, fragments

    def cal_n_step_value(self,
                         obs: List,
                         actions: List,
                         rewards: List,
                         dones: List,
                         gamma: float = 0.99,
                         n: int = 1) -> List:

        target_obs_returns = []
        reward_returns = []
        action_returns = []
        done_returns = []
        invalid_idx = []

        assert len(rewards) - n > 0
        end_idx = None

        for i in range(len(rewards) - n):
            end_idx = i
            reward = 0.
            judge_done = dones[i + 1:i + 1 + n]
            if sum(judge_done) > 1 or (sum(judge_done) == 1 and judge_done[-1] == 0):
                # reward_returns.append(None)
                invalid_idx.append(i)
                continue
            for j in range(n):
                reward += rewards[i + 1 + j] * (gamma ** j)
                end_j = j

            target_obs_returns.append(obs[i + end_j + 1])
            action_returns.append(actions[i])
            reward_returns.append(reward)
            done_returns.append(dones[i + end_j])
        x = range(len(rewards))[end_idx + 1:]
        invalid_idx.extend(x)

        return target_obs_returns, action_returns, reward_returns, done_returns, invalid_idx

    def _get_n_step(self, model_name):
        try:
            res = self.builder.models[model_name]['params']['n_step']
        except KeyError:
            res = 1
        return res

    def _get_gamma(self, model_name):
        try:
            res = self.builder.models[model_name]['params']['gamma']
        except KeyError:
            res = 0.99
        return res
