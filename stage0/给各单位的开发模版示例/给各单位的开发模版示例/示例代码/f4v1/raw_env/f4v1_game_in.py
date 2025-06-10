import random
import math


class Agent_A(object):
    # 定义A
    def __init__(self):
        # position 定义坐标
        self.x = 0
        self.y = 0
        self.d = -50  # 深度
        # status 定义状态
        self.dv = random.choice([2, -2])  # 深度速度，初始化
        self.v = random.uniform(15.433, 25.722)  # 平面速度，初始化
        self.theta = random.uniform(0, 2 * 3.1415)  # 移动角度，初始化
        self.alpha = 0  # 加速度
        self.theta_max = (-0.0657 * self.v ** 2 + 3.5332 * self.v - 7.3176) / (57.3)  # 弧度变化最大值
        self.alpha_max = self.theta_max * self.v  # 加速度最大值
        self.live = 1  # 存活

    def update_pos(self, decision_steps):
        # 更新坐标
        x_temp = self.x + math.cos(self.theta) * self.v * decision_steps
        y_temp = self.y + math.sin(self.theta) * self.v * decision_steps
        d_temp = self.d + self.dv * decision_steps
        if x_temp ** 2 + y_temp ** 2 <= 3000 ** 2:
            # 限制平面范围
            self.x = x_temp
            self.y = y_temp
        if d_temp <= 0 and d_temp >= -240:
            # 限制深度范围
            self.d = d_temp
        elif d_temp > 0:
            self.d = 0
        elif d_temp < -240:
            self.d = -240

    def get_pos(self):
        # 获取坐标
        return [int(self.x), int(self.y), int(self.d)]

    def update_theta_max(self):
        # 更新弧度最大变化值
        self.theta_max = (-0.0657 * self.v ** 2 + 3.5332 * self.v - 7.3176) / (57.3)

    def update_alpha_max(self):
        # 更新加速度最大值
        self.alpha_max = self.theta_max * self.v

    def update_status(self, X, Y, Dv):
        # 更新所有状态属性
        if abs(X / 57.3) <= self.theta_max:
            # 指令角度可达到
            self.theta = (self.theta + X / 57.3 + 2 * 3.1415) % (2 * 3.1415)
        else:
            # 指令角度无法达到，转至可达到最大角度
            self.theta = (self.theta + self.theta_max * X / abs(X) + (2 * 3.1415)) % (2 * 3.1415)
        if abs(Y - self.v) <= self.alpha_max:
            # 指令速度可达到
            self.v = Y
        else:
            # 指令速度无法达到，加速或减速至可达到最大（最小）速度
            self.v = self.v + self.alpha_max * (Y - self.v) / abs(Y - self.v)
        if self.v < 15.433:
            # 限制速度下限
            self.v = 15.433
        if self.v > 25.722:
            # 限制速度上限
            self.v = 25.722
        self.dv = Dv  # 更新深度速度
        if self.dv < -2:
            # 限制深度速度下限
            self.dv = -2
        if self.dv > 2:
            # 限制深度速度上限
            self.dv = 2
        self.update_alpha_max()  # 更新加速度范围
        self.update_theta_max()  # 更新转角范围

    def get_see(self, vis):
        # 获取探测视野示意目标点
        see_x = vis * math.cos(self.theta) + self.x
        see_y = vis * math.sin(self.theta) + self.y
        see_d = self.d
        return [int(see_x), int(see_y), int(see_d)]

    def get_status(self):
        # 获取状态值
        return [self.theta, self.v, self.alpha_max, self.theta_max, self.dv]

    def set_live(self, live):
        self.live = live

    def get_live(self):
        return [self.live, ]


class Agent_B(object):
    # 定义B，相应概念同A，只不过各项数值范围有所不同，且没有探测属性。
    def __init__(self):
        # position
        self.x = random.uniform(-2000, 2000)
        self.y = random.uniform(-1 * (2000 ** 2 - self.x ** 2) ** 0.5, (2000 ** 2 - self.x ** 2) ** 0.5)
        self.d = random.uniform(-240, 0)
        # status
        self.dv = random.choice([1, -1])
        self.v = random.uniform(0, 12)
        self.theta = random.uniform(0, 2 * 3.1415)
        self.alpha = 0
        self.theta_max = (-0.000185 * self.v ** 3 - 0.006391 * self.v ** 2 + 0.11501 * self.v) / (57.3)
        self.alpha_max = self.theta_max * self.v
        self.visible = 1  # 是否可见

    def update_pos(self, decision_steps):
        x_temp = self.x + math.cos(self.theta) * self.v * decision_steps
        y_temp = self.y + math.sin(self.theta) * self.v * decision_steps
        d_temp = self.d + self.dv * decision_steps
        if x_temp ** 2 + y_temp ** 2 <= 3000 ** 2:
            # 限制平面范围
            self.x = x_temp
            self.y = y_temp
        elif x_temp ** 2 > 3000 ** 2:
            self.x = x_temp / abs(x_temp) * 3000
            self.theta = (self.theta + self.theta_max * 180 + (2 * 3.1415)) % (2 * 3.1415)
        elif y_temp ** 2 > 3000 ** 2:
            self.y = y_temp / abs(y_temp) * 3000
            self.theta = (self.theta + self.theta_max * 180 + (2 * 3.1415)) % (2 * 3.1415)
        else:
            self.theta = (self.theta + self.theta_max * 180 + (2 * 3.1415)) % (2 * 3.1415)
        if d_temp <= 0 and d_temp >= -240:
            # 限制深度范围
            self.d = d_temp
        elif d_temp > 0:
            self.d = 0
        elif d_temp < -240:
            self.d = -240

    def get_pos(self):
        return [int(self.x), int(self.y), int(self.d)]

    def update_theta_max(self):
        self.theta_max = (-0.000185 * self.v ** 3 - 0.006391 * self.v ** 2 + 0.11501 * self.v) / (57.3)

    def update_alpha_max(self):
        self.alpha_max = self.theta_max * self.v

    def update_status(self, X, Y, Dv):
        if abs(X / 57.3) <= self.theta_max:
            self.theta = (self.theta + X / 57.3 + 2 * 3.1415) % (2 * 3.1415)
        else:
            self.theta = (self.theta + self.theta_max * X / abs(X) + (2 * 3.1415)) % (2 * 3.1415)
        if abs(Y - self.v) <= self.alpha_max:
            self.v = Y
        else:
            self.v = self.v + self.alpha_max * (Y - self.v) / abs(Y - self.v)
        if self.v < 0:
            self.v = 0
        if self.v > 12:
            self.v = 12
        self.dv = Dv
        if self.dv < -1:
            self.dv = -1
        if self.dv > 1:
            self.dv = 1
        self.update_alpha_max()
        self.update_theta_max()

    def get_status(self):
        return [self.theta, self.v, self.alpha_max, self.theta_max, self.dv]

    def set_visible(self, visible):
        self.visible = visible

    def get_visible(self):
        return [self.visible, ]
