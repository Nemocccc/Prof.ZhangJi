from drill.pipeline.interface import ObsData, ActionData, History
from drill.keys import REWARD


def reward_handler(data: ObsData, history: History):
    extra_info_dict = data.extra_info_dict
    if (extra_info_dict is None) or (REWARD not in extra_info_dict) or (not extra_info_dict):
        return {REWARD: 1.}
    else:
        return {REWARD: extra_info_dict[REWARD]}


def action_handler(data: ActionData, history: History) -> ActionData:
    return data


def feature_handler(data: ObsData, history: History):
    if isinstance(data.obs, tuple):
        name2feature = {"spatial": {"frames": data.obs[0]}}
    else:
        name2feature = {"spatial": {"frames": data.obs}}
    return name2feature
