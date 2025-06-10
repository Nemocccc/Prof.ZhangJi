from typing import Dict

import tensorflow as tf
from tensorflow import keras

from drill.keys import ACTION, ADVANTAGE, DECODER_MASK, LOGITS


class DuelingDQNNetwork(keras.Model):
    def __init__(self, action_num):
        super(DuelingDQNNetwork, self).__init__()
        self.action_num = action_num
        self._q_net = keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
        ])

        self._v_head = keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1)
        ])

        self._a_head = keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(units=action_num)
        ])

    def call(self, inputs_dict: Dict[str, tf.Tensor]):
        h = self._q_net(inputs_dict["spatial"])
        advantage = self._a_head(h)
        value = self._v_head(h)
        mean_adv = tf.reduce_mean(advantage, axis=-1, keepdims=True)
        q_values = value + advantage + mean_adv

        predict_output_dict = {
            LOGITS: {
                "action1": q_values,
            },
        }
        return predict_output_dict
