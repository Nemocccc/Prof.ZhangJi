from typing import Any, Dict, Union

import tensorflow as tf

from drill.keys import ACTION, LOGITS, REWARD, DONE
from drill.utils import get_hvd
from lib.core.drill_model.vb import VBModel
from lib.core.keys import OBS, OBS_NEXT


class DQNModel(VBModel):
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
                 gradient_clipping: Union[float, None] = 40.,
                 epsilon: dict = {"value": 1.0, "min": 0.01, "decrease": 0.000009},
                 action_names: list = [],
                 ):
        super().__init__(network, target_network, gamma, n_step, buffer_size, batch_size,
                         learning_starts, target_update_interval, training_times, optimizer_config,
                         gradient_clipping)
        self.epsilon = epsilon
        self.action_names = action_names

        self.double = False
        self.dueling = False
        self.noisy = False
        self.per = False
        self.distributional = False

    @tf.function
    def predict(self, state_dict: Dict[str, Any], epsilon) -> Dict[str, Any]:
        """ `Agent` 根据 `Env` 返回的 state, 做出决策

        Parameters
        ----------
        state_dict : Dict[str, Any]
            包括 state, reward, hidden_state 等信息，例如:
            ```python
            {
                "spatial": np.ndarray,
                "entity": np.ndarray,
                "reward": array,
                "hidden_state": Any,
                ...
            }
            ```
        epsilon : Tensor

        Returns
        -------
        Dict[str, Any]
            predict_output_dict, 包括 logits, action, value, hidden_state
        """
        predict_output_dict = self.network(state_dict)
        logits = predict_output_dict[LOGITS]
        action_dict = {}
        for action_name in self.action_names:
            action = self.argmax_sample(logits[action_name], epsilon)
            action_dict[action_name] = action
        return {ACTION: action_dict}

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
            predict_output_dict = self.network(inputs_dict[OBS])
            predict_output_dict_target = self.target_network(next_inputs_dict[OBS])

            loss = 0
            for k, v in predict_output_dict[LOGITS].items():
                value = tf.gather(v, tf.reshape(action_index[k], [-1, 1]), axis=1, batch_dims=1)
                value = tf.squeeze(value)
                target_value = tf.reduce_max(predict_output_dict_target[LOGITS][k], axis=-1)
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

    @tf.function
    def argmax_sample(self, logits, epsilon) -> tf.Tensor:
        action = tf.argmax(logits, axis=-1, output_type=tf.int32)
        batch_size = tf.shape(logits)[0]
        num_actions = tf.shape(logits)[-1]
        random_actions = tf.random.uniform(tf.stack([batch_size]), minval=0,
                                           maxval=num_actions, dtype=tf.int32)
        chose_random = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=1,
                                         dtype=tf.float32) < epsilon
        action = tf.where(chose_random, random_actions, action)
        action = tf.stop_gradient(action)
        return action
