from lib.algo.dqn.flow_model_dqn import FlowModelDQN
from lib.algo.dqn.flow_env_dqn import FlowEnvDQN
from example.dqn.cartpole.configs.builder_config import builder

algo = {'flow_env': FlowEnvDQN, 'flow_model': FlowModelDQN}

flow_config = {
    'algorithm': algo,
    'builder': builder,
    'actor_config': {
        'actor_0': {
            'training_models': [
                {
                    'model_name': 'atari_model',
                    'fragment_size': 1 + 1,
                    'replay_size': 1,
                    'sample_batch_size': 1,
                },
            ],
            'inference_models': None,
            'env_num': 10,
            'extra_info': {'index': 'training', 'description': 'used for training'},
        },
    },
}
