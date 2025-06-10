from drill.flow.flow_env_ppo import FlowEnvPPO
from drill.flow.flow_model_ppo import FlowModelPPO
from configs.builder_config import builder

algo = {'flow_env': FlowEnvPPO, 'flow_model': FlowModelPPO}

flow_config = {
    'algorithm': algo,      # 选择算法，目前最佳实践标准化流程推荐PPO
    'builder': builder,     # 选择builder，配置builder_config实现的builder
    'actor_config': {
        # 根据需求可配置不同actor进行采样，不同actor可以用于单独训练某些模型或区分"训练-验证"环境
        'actor_0': {
            # 配置需要训练的模型，多个模型通过列表方式导入
            'training_models': [
                {
                    'model_name': 'f4v1_model',     # 模型名称，注意跟network_config中对应
                    'fragment_size': 1024,          # GAE的切片大小，n-steps模式；若使用纯episode模式，需保证该值超过单局步长
                    'replay_size': 1,               # 若存在lstm、gru等时序网络，该数值可适度调大，默认1即可
                    'sample_batch_size': 128,       # 训练模型的batch_size，通常推荐为128/256/512
                    # 'max_data_reuse': 1,          # 在收集到下一个batch的数据前，最多对当前batch额外重复训练多少次, 设置为0即禁止重复使用
                    # 'putback_replays': False,       # 是否将使用的replay原路放回ReplayBuffer
                    # 'sample_mode': "LIFO",          # LIFO: 后进先出, RANDOM: 优先选择最新的未曾使用过的样本
                    # 'replay_buffer_size': 16,       # buffer中的batch_size数量上限，LIFO模式默认为16，RANDOM模式下默认为64。
                },
            ],
            'inference_models': None,               # 配置只跑前向的模型，inference_models列表中的模型仅进行模型推理，不参与训练
            # 若episode_mode为True，计算gae逻辑由nsteps（fragment_size）模式转为episode模式；需要注意的是，若使用episode模式，切片replay_size存在丢弃数据可能性。
            'episode_mode': False,
            'env_num': 20,                          # 仿真环境数量
            # extra_info中增加actor描述，extra_info支持字典格式对描述信息进行定义，可通过区分extra_info来实现"训练-验证"相独立
            'extra_info': {'index': 'training', 'description': 'used for training'},
        },
        'actor_1': {
            'training_models': None,    # 'training_models'为空，则该actor只跑前向，不训练智能体
            'inference_models': ['f4v1_model'],
            'env_num': 5,
            'extra_info': {'index': 'evaluating'},
        },
    },
}