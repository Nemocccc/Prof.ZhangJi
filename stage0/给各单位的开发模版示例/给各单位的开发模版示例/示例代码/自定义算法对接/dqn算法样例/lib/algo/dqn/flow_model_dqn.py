from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
import tensorflow as tf
import tree

from drill import summary
from drill.keys import DECODER_MASK, ACTION, DONE, REWARD

if TYPE_CHECKING:
    from drill.builder import Builder

from lib.core.flow_model.vb import FlowModelVB
from lib.utils.replay_buffers import ReplayBuffer, PrioritizedReplayBuffer
from lib.utils.schedules import LinearSchedule
from lib.core.keys import OBS, OBS_NEXT, TD_ERROR, IDXES
from lib.algo.noisydqn.utils import sample_noise, set_noisy_layer_eval


class FlowModelDQN(FlowModelVB):
    """

    Attributes
    ----------
    model_name : str
        model 的名字
    builder: Builder
        详见 `drill.builder.Builder`
    """

    def __init__(self, model_name: str, builder: Builder):
        super().__init__(model_name, builder)
        self._init(model_name, builder)

    def _init(self, model_name, builder: Builder):
        super()._init(model_name, builder)

        # 添加经验池
        if self._model.per is True:
            self.replay_buffer = PrioritizedReplayBuffer(self._model.buffer_size, self._model.alpha)
            self.beta_schedule = LinearSchedule(self._model.beta_anneal_iterations,
                                                initial_p=self._model.beta_init,
                                                final_p=self._model.beta_final)
        else:
            self.replay_buffer = ReplayBuffer(self._model.buffer_size)
            self.beta_schedule = None

    def learn(self, piece: List[Dict[str, Any]]) -> bool:
        """ `FlowModel` 使用批量数据 piece 进行学习，训练模型

        Parameters
        ----------
        piece : List[Dict[str, Any]]
            由 state_dict, behavior_info_dict, decoder_mask, advantage 组成。

            * state_dict 包含 state， reward， done 等信息，还可能包含 hidden_state;
            * behavior_info_dict 包含 logits, action, value;
            * decoder_mask 包含 valid action。

        Returns
        -------
        bool
            是否将数据推送给 PredictorService
        """
        super().learn(piece)

        state_dict, behavior_info_dict, mask_dict, advantage = piece

        behavior_info_dict[DECODER_MASK] = mask_dict[DECODER_MASK]

        state_dict_ = state_dict[OBS]
        action_dict = behavior_info_dict[ACTION]
        reward = state_dict[REWARD]
        next_state_dict = state_dict[OBS_NEXT]
        done = state_dict[DONE]

        for value in state_dict_.values():
            length = value.shape[0]
            break
        for i in range(length):
            self.replay_buffer.add(tree.map_structure(lambda x: x[i], state_dict_),
                                   tree.map_structure(lambda x: x[i], action_dict),
                                   reward[i],
                                   tree.map_structure(lambda x: x[i], next_state_dict),
                                   done[i])

        summary.scalar('buffer_size', len(self.replay_buffer), step=self.learn_step)

        if len(self.replay_buffer) > self._model.learning_starts:
            for _ in range(self._model.training_times):
                if self._model.per is True:
                    state_dict, behavior_info_dict = self.replay_buffer.sample(
                        self._model.batch_size,
                        beta=self.beta_schedule.value(self.learn_step)
                    )
                    idxes = state_dict.pop(IDXES)
                else:
                    state_dict, behavior_info_dict = self.replay_buffer.sample(
                        self._model.batch_size
                    )

                if self._model.noisy is True:
                    sample_noise(self._model.network)
                    sample_noise(self._model.target_network)

                summary_dict = self._model.learn(state_dict, behavior_info_dict)

                if self._model.per is True:
                    # 更新per-buffer权重
                    td_error = summary_dict.pop(TD_ERROR)
                    new_priorities = np.abs(td_error) + self._model.prioritization_epsilon
                    self.replay_buffer.update_priorities(idxes, new_priorities)

                for k, v in summary_dict.items():
                    summary.scalar(k, v, step=self.learn_step)

            if self.learn_step % self._model.target_update_interval == 0:
                weights_ = self._model.network.get_weights()
                self._model.target_network.set_weights(weights_)

        return True

    def predict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """ `FlowModel` 进行前向预测

        Parameters
        ----------
        state_dict : Dict[str, Any]
            模型 inference 的输入

        Returns
        -------
        Dict[str, Any]
            模型 inference 的结果

        Examples
        --------
        state_dict
        ``` python
        {
            "spatial": np.ndarray,
            "entity": np.ndarray,
            "reward": array,
            "hidden_state": Any,
            ...
        }
        ```

        return
        ```python
        {
            "logits": {
                "x": np.ndarray,
                "y": np.ndarray
            },
            "action": {
                "x": np.ndarray,
                "y": np.ndarray
            }
            "value": np.ndarray,
            "hidden_state": np.ndarray
        }
        ```
        """
        value = self._model.epsilon['value'] - self._model.epsilon['decrease']
        self._model.epsilon['value'] = max(value, self._model.epsilon['min'])
        summary.scalar('epsilon_predict', self._model.epsilon['value'])

        if self._model.noisy is True:
            # todo 本地前向时手动开启eval，待优化（需修改drill local）
            # set_noisy_layer_eval(self._model.network)
            sample_noise(self._model.network)

        predict_output_dict = self._model.predict(state_dict,
                                                  tf.constant(self._model.epsilon['value']))
        output_dict = tree.map_structure(lambda x: x.numpy(), predict_output_dict)
        return output_dict
