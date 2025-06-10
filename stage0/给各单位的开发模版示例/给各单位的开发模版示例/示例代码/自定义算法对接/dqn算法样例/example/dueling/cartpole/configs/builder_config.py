from drill.pipeline.agent_pipeline import AgentPipeline, HandlerSpecies
from example.dueling.cartpole.configs.network_config import model_config, spatial_feature_set
from example.dueling.cartpole.env_interface import AtariEnv, ENV_NAME
from example.dueling.cartpole.pipeline import feature_handler, reward_handler, action_handler
from lib.algo.dqn.dqn_builder import DQNBuilder

# env 配置
env = {"class": AtariEnv, "params": {"extra_info": ENV_NAME}}

# pipeline 配置
pipeline = {
    "atari_pipeline": {
        "class": AgentPipeline,
        "params": {
            "handler_dict": {
                HandlerSpecies.FEATURE: (feature_handler, [spatial_feature_set]),
                HandlerSpecies.REWARD: reward_handler,
                HandlerSpecies.ACTION: action_handler,
            },
        },
    },
}

# agent_name to model_name
agents = {
    'atari': {
        "model": "atari_model",
        "pipeline": "atari_pipeline"
    },
}

builder = DQNBuilder(agents, model_config, env, pipeline)
