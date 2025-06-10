from drill.builder import BPBuilder
from env_interface import F4v1Env
from drill.pipeline.agent_pipeline import AgentPipeline, HandlerSpecies
from drill.pipeline import GlobalPipeline
from pipeline import feature_handler, reward_handler, action_handler, player_done_process
from configs.network_config import entity_feature_set, target_feature_set
from configs.network_config import model_config


# env 配置，如果需要对环境进行配置，可以通过params传递相应参数，并在env创建时使用
env = {"class": F4v1Env, "params": {}}

# 智能体名称
AGENT_NAMES = ['player0', 'player1', 'player2', 'player3']

# pipeline 配置
pipeline = {
    # 根据pipeline中实现的feature_handler，action_handler，reward_handler等函数配置智能体训练pipeline
    "f4v1_pipeline": {
        "class": AgentPipeline,
        "params": {
            "handler_dict": {
                HandlerSpecies.FEATURE: (feature_handler, [
                    entity_feature_set,
                    target_feature_set,
                    # x_mask_feature_set,
                ]),
                HandlerSpecies.REWARD: reward_handler,
                HandlerSpecies.ACTION: action_handler,
            },
            "batch_config": {
                "gamma": 0.99,
                "lamb": 0.95,
            }
        },
    },
    "global": {
        "class": GlobalPipeline,
        "params": {
            "pre_process": player_done_process
        }
    }
}

# 智能体与神经网络模型映射关系，支持多智能体
agents = {
    'player0': {
        "model": "f4v1_model",          # 选择智能体对应网络模型
        "pipeline": "f4v1_pipeline"     # 选择对应pipeline配置
    },
    'player1': {
        "model": "f4v1_model",
        "pipeline": "f4v1_pipeline"
    },
    'player2': {
        "model": "f4v1_model",
        "pipeline": "f4v1_pipeline"
    },
    'player3': {
        "model": "f4v1_model",
        "pipeline": "f4v1_pipeline"
    },
}

builder = BPBuilder(agents, model_config, env, pipeline)