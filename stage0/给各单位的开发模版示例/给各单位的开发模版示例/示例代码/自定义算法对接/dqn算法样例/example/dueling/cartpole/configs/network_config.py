from drill.feature import VectorFeature, SpatialFeatureSet
from example.dueling.cartpole.dqn_net import DuelingDQNNetwork
from example.dueling.cartpole.env_interface import AtariEnv, ENV_NAME
from lib.algo.dqn.dqn_model import DQNModel

temp_env = AtariEnv(0, ENV_NAME)
action_space = temp_env.action_space.n
state_space = temp_env.observation_space.shape
del temp_env

# feature 配置
spatial_feature_set = SpatialFeatureSet(name="spatial",
                                        shape=state_space[:-1],
                                        feature_dict={"frames": VectorFeature(state_space[-1])})

# network 配置
network = {
    "class": DuelingDQNNetwork,
    "params": {"action_num": action_space}
}

# model 配置
model_config = {
    "atari_model": {
        "class": DQNModel,
        "params": {
            "network": network,
            "gamma": 0.99,
            "n_step": 1,
            "buffer_size": 1e+4,
            "batch_size": 512,
            "learning_starts": 2000,
            "target_update_interval": 200,
            "epsilon": {"value": 1.0, "min": 0.01, "decrease": 0.000009},
            "optimizer_config": {"class_name": 'Adam', "config": {"learning_rate": 1e-3, }},
            # "optimizer_config": 'Adam',
            "gradient_clipping": 40.0,
            "training_times": 1,
            "action_names": ["action1"],
        },
        "save": {
            "interval": 10000,
        },
    }
}
