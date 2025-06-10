from drill.pipeline.interface import ObsData, ActionData, History


def reward_handler(data: ObsData, history: History):
    """
    实现奖励函数，并根据态势数据计算当前奖励

    :param data: 原始态势信息
    :param history: 历史信息
    :return: 智能体当前获得奖励
    """
    # 在f4v1样例中，环境返回值包括了原始奖励，则在奖励计算时，可以直接使用环境奖励
    extra_info_dict = data.extra_info_dict
    if (extra_info_dict is None) or ('reward' not in extra_info_dict) or (not extra_info_dict):
        return {'reward': 0}
    else:
        return {'reward': extra_info_dict['reward']}


def feature_handler(data: ObsData, history: History):
    """
    o2s，实现态势信息到神经网络输入信息的数据转换

    :param data: 原始态势信息
    :param history: 历史信息
    :return: 神经网络输入数据
    """
    my_units_list, ally_feature = [], []
    for k, v in data.obs.items():
        if k == 'b_info':
            b_info = {
                'b_pos': v['b_pos'],
                'b_visible': v['b_visible']
            }
            continue
        if k[-1] == data.agent_name[-1]:
            my_units_list.append(v)
        else:
            if any([item.sum() for item in v.values()]):
                ally_feature.append(v)
    my_units_list.extend(ally_feature)
    name2feature = {
        'my_units': my_units_list,
        'b_info': b_info,
        # 'x_mask': {"x_mask_inputs": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}, # mask信息也要通过feature_handler统一传输
    }
    # history.agents_history[data.agent_name] = name2feature # 如果想将obs数据传到action_handler中，可通过该方式实现
    return name2feature


def action_handler(data: ActionData, history: History) -> ActionData:
    """
    a2c，实现网络输出到智能体动作的转换
    注：data.action会用于训练，目前版本不建议a2c在该模块实现，后续版本会对该部分进行修正

    :param data: 原始神经网络输出
    :param history: 历史信息
    :return: 解析后的动作
    """
    # f4v1样例中，环境可以直接接受神经网络输出值，则不需要做动作映射处理
    # data.action.pop('action_x')   # 如果想令某个动作头不参与训练，即valid_action为false，则可以通过该方式实现
    return data


def player_done_process(data: ObsData, history: History):
    """
    在多智能体环境中，经常会出现部分训练单位在episode结束前死亡的情况。
    不同的仿真环境对死亡单位的处理方式不同，有的可能直接传递一个空的状态，有的可能传递一个全0的状态，
    有的可能保留其死亡前的状态，有的可能直接剔除死亡单位。
    为了保障数据流的正确性和智能体模型训练效果，在训练单位死亡后应当不再训练其智能体。
    因此我们需要对死亡单位的数据进行屏蔽处理。

    该方法用于实现屏蔽死亡训练单位数据的功能

    :param data: 原始态势数据
    （注：data为dict类型，包含'agent_name'、'obs'、'extra_info_dict'三个字段。
    其中'extra_info_dict'包括'episode_done'字段以及其他从Env.step中obs_dict传过来的变量，'episode_done'表示本局对抗是否结束。）
    :param history: 历史信息
    :return: 经过预处理后的态势信息，传给feature_handler
    """
    pre_agent_dict = {}
    for agent_name, info in data.items():
        # 对齐环境中单位的名字
        agent_name_fix = agent_name[:-1] + '_' + agent_name[-1]

        # 本环境中，死亡单位会传递全0的状态，我们可以通过判断状态是否全0来判断单位是否死亡
        # player_done为True表示单位死亡，为False表示单位存活
        player_done = not any([item.sum() for item in info.obs[agent_name_fix].values()])

        # 一个episode结束时，会立即获取初始状态，无需屏蔽数据，否则会导致KeyError
        # episode没结束时，屏蔽死亡训练单位的数据。
        if info.extra_info_dict.get("episode_done", False) or not player_done:
            pre_agent_dict[agent_name] = info

    return pre_agent_dict, history
