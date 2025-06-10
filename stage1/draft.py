import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, Box
from gymnasium.vector import AsyncVectorEnv

# 定义一个具有字典状态空间和动作空间的环境
class CustomEnv(gym.Env):
    def __init__(self):
        self.observation_space = Dict({
            "position": Box(low=-1, high=1, shape=(2,)),
            "color": Discrete(3)
        })
        self.action_space = Dict({
            "move": Discrete(4),
            "use_item": Discrete(2)
        })

    def reset(self, seed=None, options=None):
        return {"position": [0.0, 0.0], "color": 0}, {}

    def step(self, action):
        # 示例逻辑
        reward = 0
        done = False
        return {"position": [0.1, 0.1], "color": 1}, reward, done, False, {}


if __name__ == '__main__':
        
    # 创建向量化环境
    envs = AsyncVectorEnv([
        lambda: CustomEnv(),
        lambda: CustomEnv()
    ], shared_memory=False)

    # 测试向量化环境
    obs, info = envs.reset()
    print(obs)
    actions = [{"move": 1, "use_item": 0}, {"move": 2, "use_item": 1}]
    obs, rewards, terminations, truncations, infos = envs.step(actions)
    print(obs)