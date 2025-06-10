from basic import *
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches


# 定义雷达信号干扰模型
class JammingModel:
    def __init__(self, jam_strength=0.5):
        self.jam_strength = jam_strength  

    def interfere(self, signal_strength):
        """计算干扰后的信号质量""" 
        # 简单模型：信号强度 × (1 - 干扰强度) + 随机噪声
        return signal_strength * (1 - self.jam_strength) + np.random.normal(0, 0.1)


class BlueEnv(gym.Env):
    metadata = {"render_modes": ['human'], "render_fps": 1}

    def __init__(self, render_mode, num_fighters=2):
        super(BlueEnv, self).__init__()
        self.render_mode = render_mode

        self.num_fighters = num_fighters  # 战斗机数量
        self.jamming_model = JammingModel(jam_strength=0.3)  # 初始化干扰模型

       
        # 先考虑简单情况下的预警机动作：0-切换ETS模式，1-切换AW模式，2-切换ESS模式，3-保持当前模式
        # 还会有飞机的雷达的动作（变化侦察范围）
        self.action_space = spaces.Discrete(4)

        # 先考虑简单的二维观测空间情况：
        # [预警机位置x, 预警机位置y, 预警机雷达ERP, 预警机雷达频率,
        #  战斗机1位置x, 战斗机1位置y, 战斗机1雷达ERP, 战斗机1雷达频率,
        #  ..., 干扰后预警机信号, 干扰后战斗机信号]
        obs_dim = 4 + (4 * num_fighters) + 2  # 预警机4 + 4*每架战斗机4 + 干扰后的信号2 =22
        inf = np.inf
        self.observation_space = spaces.Box(-inf, inf, shape=(obs_dim,), dtype=np.float32)

        # 蓝方对象初始化 
        self.aircraft_list = []
        self.ew_aircraft = EWAircraft(id= 0,speed = 0)   # 预警机
        self.fighters = []  # 战斗机列表

        # 渲染相关 
        self.fig = None
        self.ax = None
        self.steps = 0
        self.max_steps = 1000

    def _initialize_blue_forces(self):
        """初始化蓝方单位"""
        # 初始化预警机（位置随机在左上区域）
        self.ew_aircraft = EWAircraft(id="EW-01", speed=200)
        self.ew_aircraft.position = np.array([-500, 500])  # 初始位置
        # 初始化战斗机（位置随机在右侧区域）
        for i in range(self.num_fighters):
            fighter = F18(id=f"F-{i + 1}", speed=300)
            fighter.position = np.array([np.random.uniform(0, 1000), np.random.uniform(-500, 500)])

            self.fighters.append(fighter)

        self.aircraft_list = [self.ew_aircraft] + self.fighters

    def _get_radar_signals(self):
        """获取所有雷达的原始信号及受干扰后的信号"""
        raw_signals = []
        jammed_signals = []

        # 预警机雷达信号
        ew_raw = self.ew_aircraft.radar.erp
        ew_jammed = self.jamming_model.interfere(ew_raw) #将信号功率传入，返回受影响后的信号功率
        raw_signals.append(ew_raw)
        jammed_signals.append(ew_jammed)

        # 战斗机雷达信号
        for fighter in self.fighters:
            fr_raw = fighter.radar.erp
            fr_jammed = self.jamming_model.interfere(fr_raw)
            raw_signals.append(fr_raw)
            jammed_signals.append(fr_jammed)

        return raw_signals, jammed_signals

    def _get_obs(self):
        """生成观测值"""
        #reset时初始化蓝方各单位
        if not self.aircraft_list:
            self._initialize_blue_forces() 

        obs = []

        # 添加预警机状态（位置、雷达参数） 2+1+1 = 4
        obs.extend(self.ew_aircraft.position)
        obs.append(self.ew_aircraft.radar.erp)
        obs.append(self.ew_aircraft.radar.frequency)

        # 添加战斗机状态 4*（2+1+1） = 16
        for fighter in self.fighters: 
            obs.extend(fighter.position)
            obs.append(fighter.radar.erp)
            obs.append(fighter.radar.frequency)

        # 添加干扰后的信号（预警机+战斗机汇总）2
        _, jammed_signals = self._get_radar_signals()
        obs.extend(jammed_signals)

        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.aircraft_list = []
        self.ew_aircraft = None
        self.fighters = []
        return self._get_obs(), {}

    def step(self, action):
        # 处理预警机动作（切换雷达模式）
        if action == 0:
            self.ew_aircraft.radarControl("ETS")
        elif action == 1:
            self.ew_aircraft.radarControl("AW")
        elif action == 2:
            self.ew_aircraft.radarControl("ESS")
        # action=3为保持当前模式，无需操作
        elif action == 3:
            pass

        # 执行飞机移动
        self.ew_aircraft.position += np.array([5, 0])  # 预警机向右移动
        for fighter in self.fighters:
            fighter.position += np.random.normal(0, 10, size=2)  # 随机扰动

        # 计算奖励
        raw_signals, jammed_signals = self._get_radar_signals()
        signal_stability = np.mean([np.abs(rs - js) for rs, js in zip(raw_signals, jammed_signals)])
        reward = -signal_stability  # 信号干扰越大，奖励越低

        # 步数管理
        self.steps += 1
        truncated = self.steps >= self.max_steps
        done = False  

        return self._get_obs(), reward, done, truncated, {}

    def render(self):
        if self.render_mode != "human":
            return None

        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            plt.show(block=False)

        self.ax.clear()

        # ✅ 强制每次设置坐标轴
        self.ax.set_xlim(-1000, 1000)
        self.ax.set_ylim(-1000, 1000)
        self.ax.set_aspect('equal')

        # 绘制预警机
        ew_pos = self.ew_aircraft.position
        self.ax.add_patch(patches.RegularPolygon(
            (ew_pos[0], ew_pos[1]), 3, radius=20, orientation=np.radians(90),
            color='red', alpha=0.8, ec='black'
        ))
        print(f"EW Position: {ew_pos}")

        # 绘制战斗机
        for fighter in self.fighters:
            self.ax.add_patch(patches.Circle(
                (fighter.position[0], fighter.position[1]), 15,
                color='blue', alpha=0.8, ec='black'
            ))
            print(f"fighter Position: {fighter.position}")

        self.ax.text(-900, 900, f"Step: {self.steps}", bbox=dict(facecolor='white', alpha=0.8))
        plt.pause(0.5)

    def close(self):
        if self.fig:
            plt.close(self.fig)



if __name__ == "__main__":
    env = BlueEnv(render_mode='human', num_fighters=3)
    obs, _ = env.reset()
    for _ in range(100):
        action = env.action_space.sample()  # 随机动作
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        print(f"Step: {env.steps}, Obs shape: {obs.shape}, Reward: {reward:.2f}")
        if done or truncated:
            break
    env.close()