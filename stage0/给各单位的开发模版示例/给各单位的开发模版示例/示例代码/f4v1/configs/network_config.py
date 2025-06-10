from drill.feature import VectorFeature, PlainFeature, OnehotFeature, RangedFeature
from drill.feature import CommonFeatureSet, EntityFeatureSet
from drill.model.tf.network.encoder import EntityEncoder, CommonEncoder, encoder_pooling
from drill.model.tf.network.aggregator import GRUAggregator, DenseAggregator, LSTMAggregator
from drill.model.tf.network.decoder import CategoricalDecoder, SingleSelectiveDecoder, UnorderedMultiSelectiveDecoder, \
    OrderedMultipleSelectiveDecoder, GaussianDecoder
from drill.model.tf.network.layer import ValueApproximator
from drill.model.tf.network.commander import CommanderNetwork
from drill.model.tf import CommanderModelPPO

"""
常用特征模板
1.common_feature_set = CommonFeatureSet(name: str, feature_dict: dict)
    通用特征模版。作为相应数据类型的载体，用于提取通用特征。
    通用特征的特点是数据定长，数据输入的 shape 固定。

2.EntityFeatureSet(name: str, max_length: int, feature_dict: dict)
    实体特征模版。用于提取实体特征信息。
    实体特征不同于通用特征，其长度可变。在推演中由于毁伤或不完全观测等原因，导致实体信息形状不固定；而本特征模版则用于处理这类的变长信息。

常用特征
1.PlainFeature：
    单数据特征类型。用于表示单个连续的数据。
    例如: 一些经度、纬度、高度、血量、距离、都可以用此种特征表示。
2.VectorFeature(length：int)：
    向量特征类型。用于处理特征信息。在推演运行时，本类定义的特征类型支持向量输入，输入向量长度需要和初始化参数 length 相同
    例如：高纬的坐标信息。
3.RangedFeature(low: float, high: float, length: int) 
    用于处理需要归一化的特征信息。在推演运行时，本类定义的特征类型支持单数据和向量类型输入。输入后会统一对数据做归一化处理。
    归一化处理方式是将每一个数据 x 都进行（x-最小值）/（最大值-最小值）的处理
4.OnehotFeature(depth: int)
    用于处理类别信息，可以将一个整数转换为一个 one-hot 的表示向量。
    例如：男女性别、存活状态。
"""

# feature 配置
entity_feature_set = EntityFeatureSet(
    name='my_units',
    max_length=4,
    feature_dict={
        "pos": VectorFeature(3),  # 坐标信息
        'theta': PlainFeature(),  # 转角信息
        'v': PlainFeature(),  # 二维平面速度
        'alpha_max': PlainFeature(),  # alpha转角限制
        'theta_max': PlainFeature(),  # theta转角限制
        'dv': PlainFeature(),  # 垂直（深度）速度
        'radar': VectorFeature(3),  # 雷达射线正前方最远点坐标信息
    },
)
target_feature_set = CommonFeatureSet(
    name='b_info',
    feature_dict={
        'b_pos': VectorFeature(3),  # 坐标信息
        'b_visible': PlainFeature(),  # 是否可见，该类信息可根据场景调整为onehot型特征
    }
)

# # 如果想使用action_mask，在这里加入一个CommonFeatureSet
# x_mask_feature_set = CommonFeatureSet(
#     name="x_mask",
#     feature_dict={
#         "x_mask_inputs": VectorFeature(10)  # mask使用时的纬度，要跟动作头纬度对应
#     }
# )

# feature输入
encoders = {
    "entity_encoder": {
        "class": EntityEncoder,
        "params": {
            "hidden_layer_sizes": [256, 128],  # 隐藏层参数
            "transformer": None,  # transformer参数
            "pooling": encoder_pooling.Max(),  # 池化方法，可选 Max() 或 Attention(num_query, num_head, head_size)
        },
        "inputs": entity_feature_set  # 输入特征配置
    },
    "common_encoder": {
        "class": CommonEncoder,
        "params": {
            "hidden_layer_sizes": [256, 128],
        },
        "inputs": target_feature_set
    }
}

# feature聚合器参数配置，通常可选用GRUAggregator或DenseAggregator
aggregator = {
    "class": GRUAggregator,
    "params": {
        "hidden_layer_sizes": [512, 256],
        "state_size": 64,
        "output_size": 512,
        "seq_len": 1,
    }
}
# aggregator = {
#     "class": DenseAggregator,
#     "params": {
#         "hidden_layer_sizes": [512, 256],
#         "output_size": 512,
#     }
# }

# action输出
decoders = {
    "action_x": {
        "class": CategoricalDecoder,  # 离散动作选择
        "params": {
            "n": 10,  # 动作数量
            "hidden_layer_sizes": [512, 256],  # 解码器隐藏层参数
            # "activation": 'relu',
            # "temperature": 1.0,
        },
        # "mask": 'x_mask',                     # 如果使用action_mask，增加mask参数
        # "dependency": None,
    },
    "action_y": {
        "class": CategoricalDecoder,
        "params": {
            "n": 10,
            "hidden_layer_sizes": [512, 256]
        }
    },
    "action_dv": {
        "class": CategoricalDecoder,
        "params": {
            "n": 10,
            "hidden_layer_sizes": [512, 256]
        }
    }
}

# critic参数配置
value_layer = {
    "class": ValueApproximator,
    "params": {
        "hidden_layer_sizes": [64, 32]
    }
}

# network 配置，将定义好的编码器、聚合器、解码器、价值评估网络进行组合
network = {
    "class": CommanderNetwork,
    "params": {
        "encoder_config": encoders,
        "aggregator_config": aggregator,
        "decoder_config": decoders,
        "value_approximator_config": value_layer,
    }
}

# model 配置
model_config = {
    "f4v1_model": {
        "class": CommanderModelPPO,  # 选用最佳实践推荐的模型，基于ppo的CommanderModel
        "params": {
            "network": network,  # 神经网络结构
            # "sync": False,            # 2152：是否开启同步模式，默认为False，即异步训练；True则为同步训练
            # "sync_interval": 10,      # 2152：若选择同步训练模式，则需设定模型同步间隔，默认为10次权重更新-同步模型1次（清空buffer）
            "learning_rate": 2e-4,  # 学习率
            "clip_param": 0.3,  # 为了训练稳定，限制新旧 policy network 的差距
            "vf_clip_param": 10.,  # 为了训练稳定，限制 value network 的差距
            "vf_loss_coef": 1.,  # value loss 的 scale factor（影响因子）, 为 1 则不 scale
            "entropy_coef": 0.1,  # entropy loss 的 scale factor（影响因子）, 为 1 则不 scale
        },
        # 模型存储参数
        "save": {
            "interval": 100,  # 模型存储间隔，即网络更新多少次存储一次模型
        },
        # 模型加载参数，不设置则默认不进行预训练模型加载
        # "load": {
        #     "model_file": "models/f4v1x_model/f4v1x_model_200.npz",   # 预训练模型存储路径
        # }
    },
}
