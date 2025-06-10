from typing import Dict

import tensorflow as tf
from tensorflow import keras

from drill.keys import ACTION, ADVANTAGE, DECODER_MASK, LOGITS


class DQNNetwork(keras.Model):

    def __init__(self, action_num):
        super(DQNNetwork, self).__init__()
        self.action_num = action_num
        self._q_net = keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(units=action_num)
        ])

    def call(self, inputs_dict: Dict[str, tf.Tensor]):
        q_values = self._q_net(inputs_dict["spatial"])

        predict_output_dict = {
            LOGITS: {
                "action1": q_values,
            },
        }
        return predict_output_dict
