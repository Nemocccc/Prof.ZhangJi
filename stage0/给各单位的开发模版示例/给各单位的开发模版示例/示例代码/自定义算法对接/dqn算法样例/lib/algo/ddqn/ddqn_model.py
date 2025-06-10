from typing import Any, Dict, Union

import tensorflow as tf

from drill.keys import ACTION, LOGITS, REWARD, DONE
from drill.utils import get_hvd
from lib.algo.dqn.dqn_model import DQNModel
from lib.core.keys import OBS, OBS_NEXT


class DDQNModel(DQNModel):
    """Drill 根据 RL 的使用场景提供的一个 Model 的实现
    """

    def __init__(self,
                 network,
                 target_network,
                 gamma: float = 0.99,
                 n_step: int = 1,
                 buffer_size: int = 1e+5,
                 batch_size: int = 1024,
                 learning_starts: int = 2000,
                 target_update_interval: int = 200,
                 training_times: int = 1,
                 optimizer_config: Union[str, dict] = {"class_name": 'Adam',
                                                       "config": {"learning_rate": 1e-3, }},
                 gradient_clipping: Union[float, None] = 40.0,
                 epsilon: dict = {"value": 1.0, "min": 0.01, "decrease": 0.000009},
                 action_names: list = [],
                 ):
        super().__init__(network, target_network, gamma, n_step, buffer_size, batch_size,
                         learning_starts, target_update_interval, training_times, optimizer_config,
                         gradient_clipping, epsilon, action_names)

        self.double = True

    @tf.function
    def learn(self,
              inputs_dict: Dict[str, Any],
              behavior_info_dict: Dict[str, Any]
              ) -> Dict[str, Any]:
        """使用与环境交互获得的 rollout 数据来进行训练

        Parameters
        ----------
        inputs_dict : Dict[str, Any]
            包含 state, reward, hidden_state 等信息，例如:
            ```python
            {
                "spatial": np.ndarray,
                "entity": np.ndarray,
                "reward": array,
                "hidden_state": Any,
                ...
            }
            ```
        behavior_info_dict : Dict[str, Any]
            一个包含 logits, action, value, advantage, decoder_mask 的
            Dict, 是 behavior network 的输出。

        Returns
        -------
        Dict[str, Any]
            训练过程中产生的一些统计数据，比如 loss, entropy, kl 等
        """

        reward = inputs_dict[REWARD]
        action_index = behavior_info_dict[ACTION]
        dones = inputs_dict[DONE]
        next_inputs_dict = {OBS: inputs_dict[OBS_NEXT]}

        with tf.GradientTape() as tape:
            # 输入 state, reward, behavior action; 输出 value, new action, new logits（new policy）；
            predict_output_dict = self._network(inputs_dict[OBS])
            predict_output_dict_target = self._target_network(next_inputs_dict[OBS])

            loss = 0
            for k, v in predict_output_dict[LOGITS].items():
                value = tf.gather(v, tf.reshape(action_index[k], [-1, 1]), axis=1, batch_dims=1)
                value = tf.squeeze(value)
                selected_action = tf.argmax(v, axis=1)
                target_value = tf.gather(predict_output_dict_target[LOGITS][k],
                                         tf.reshape(selected_action, [-1, 1]),
                                         axis=1, batch_dims=1)
                target_value = tf.squeeze(target_value)
                target = reward + self.gamma ** self.n_step * (1. - dones) * target_value
                loss += tf.reduce_mean(tf.square(tf.stop_gradient(target) - value) * 0.5)

        hvd = get_hvd("tensorflow")
        tape = hvd.DistributedGradientTape(tape)
        # 反向传播，计算所有 weight 的 gradient
        gradients = tape.gradient(loss, self._network.trainable_variables)
        if isinstance(self.gradient_clipping, float):
            gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)
        # 更新 weight
        self._optimizer.apply_gradients(zip(gradients, self._network.trainable_variables))

        summary = {
            "loss": loss,
            "average_q": tf.reduce_mean(value)
        }
        return summary
