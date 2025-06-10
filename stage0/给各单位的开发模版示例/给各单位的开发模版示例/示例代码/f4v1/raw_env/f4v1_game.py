import copy
import os
import time
import numpy as np
from raw_env.env_def import *
from raw_env.f4v1_game_in import *

if RENDER is True:
    from raw_env.draw_env import *
os.environ["SDL_VIDEO_WINDOW_POS"] = "%d, %d" % (10, 30)


class Env(object):
    # 定义环境，初始化A和B
    def __init__(self, title=None):
        if RENDER is True:
            pygame.init()
            pygame.font.init()
            self.display_surf = pygame.display.set_mode((WIDTH, HEIGHT + DEEPHEIGHT))
            self.abplayer = DisplayPlayer(self.display_surf)
            self.display_surf.fill((0, 191, 255))
        self.vis = vis0  # 初始化探测距离
        self.running = True  # 运行判断符
        self.rew = [0, 0, 0, 0]
        self.abdistance = [[float('inf'), float('inf')], [float('inf'), float('inf')],
                           [float('inf'), float('inf')], [float('inf'), float('inf')]]
        self.hit = [False, False, False, False]
        self.hit_num = 0
        self.step_cum = 0
        self.find_step = 0
        self.find = False
        self.ob_done = False
        self.hit_v = [0, 0, 0, 0]
        self.ob_info = {'eprew': 0, 'eplen': 0, 'pldone': self.hit, 'final_v': self.hit_v}
        self.A = []  # 智能体A种群初始化
        for i in list(range(4)):
            # 初始化A
            self.A.append(Agent_A())
        self.B = Agent_B()  # 初始化B

    def step(self, choose_action0, vis_steps):
        # 控制A和B运动以及刷新环境，
        if self.find is False:
            objects = [0, ]
        else:
            objects = [1, ]
        log = []
        self.ob_info['eplen'] = self.ob_info['eplen'] + 1
        self.step_cum = self.step_cum + 1
        self.rew = [0, 0, 0, 0]
        for i in list(range(4)):
            if self.hit[i] is False:
                self.A[i].update_pos(vis_steps)
        self.B.update_pos(vis_steps)  # 更新B的位置
        for i in list(range(4)):
            if self.hit[i] is False:
                ax0 = choose_action0[i][0]['action_x']
                bx0 = choose_action0[i][0]['action_y']
                cx0 = choose_action0[i][0]['action_dv']
                if DISCRETE_ACTION:
                    AX = -180 + 360 * ax0 / 9
                    AY = 15 + 13 * bx0 / 9
                    ADv = -2 + 4 * cx0 / 9
                else:
                    # 连续动作默认为-1到1
                    if any([x < -1 or x > 1 for x in [ax0, bx0, cx0]]):
                        raise ValueError('Action should be in [-1, 1]!')
                    AX = 180 * ax0
                    AY = 21.5 + 6.5 * bx0  # AY \in (15, 28)
                    ADv = -2 * cx0
                self.A[i].update_status(-AX, AY, ADv)  # X负值因为数学模型中定义X正为向右，与三角函数计算方式有所不同
            BX = random.uniform(-180, 180)
            BY = random.uniform(0, 12)
            BDv = random.uniform(-1.5, 1.5)
            self.B.update_status(-BX, BY, BDv)
        np_ob0 = []
        for i in list(range(4)):
            # 将A的位置信息、状态信息和预测信息放到信息列表中
            np_ob0 = np_ob0 + self.A[i].get_pos() + self.A[i].get_status() + self.A[i].get_see(self.vis)
            objects.append(
                {'A_' + str(i) + '_position': self.A[i].get_pos(), 'A_' + str(i) + '_live': self.A[i].get_live(),
                 'A_' + str(i) + '_status': self.A[i].get_status(),
                 'A_' + str(i) + '_see': self.A[i].get_see(self.vis)})
            tempa = self.A[i].get_status()
            tempax = tempa[0:2]
            tempax.append(tempa[4])
            log.append(
                {'A_' + str(i) + '_position': self.A[i].get_pos(),
                 'A_' + str(i) + '_status': tempax})
        objects.append({'B_position': self.B.get_pos(),
                        'B_status': self.B.get_status(),
                        })  # 将B的位置信息、状态信息放到信息列表中
        tempb = self.B.get_status()
        tempbx = tempb[0:2]
        tempbx.append(tempb[4])
        log.append({'B_position': self.B.get_pos(), 'B_status': tempbx})
        if self.find is False:
            objects[0] = self.find_tar()
        if self.find is True:
            np_ob0 = np_ob0 + self.B.get_pos() + self.B.get_visible()
            self.hit_tar()
        else:
            np_ob0 = np_ob0 + [0, 0, 0, 0]
        self.hit_num = sum(self.hit)
        if self.hit_num == 4:
            self.ob_done = True
        np_ob = np.asarray(np_ob0)
        np_ob = np_ob.reshape((-1))  # pos=3,status=5,see=3,*4   bpos=3,visible=1
        players_np_ob = self.players_ob(np_ob)
        players_np_ob_considerlose = self.players_ob_lose(players_np_ob)
        if RENDER is True:
            self.display_surf.fill((0, 191, 255))
            if self.ob_done is True:
                objects[0] = 10
            self.abplayer.update(objects)
            pygame.display.update()
            if WRITELOG:
                filenamex = './f4v1-x5.log'
                with open(filenamex, 'a+') as fw:
                    fw.write(str(log))
                    fw.write('\n')
                    if self.ob_done is True:
                        fw.write('\n\n--------------------------\n\n')
            time.sleep(.1)
            if self.ob_done is True:
                time.sleep(1)
        self.ob_info['pldone'] = copy.copy(self.hit)
        self.ob_info['final_v'] = copy.copy(self.hit_v)
        if self.step_cum >= MAXSTEPS:
            self.ob_done = True
        env_obs = {'player0': dict(), 'player1': dict(), 'player2': dict(), 'player3': dict(), }
        for i in list(range(4)):
            for j in list(range(4)):
                env_obs['player' + str(i)]['player_' + str(j)] = {
                    'pos': players_np_ob_considerlose[i][j * 11 + 0:j * 11 + 3],
                    'theta': players_np_ob_considerlose[i][j * 11 + 3],
                    'v': players_np_ob_considerlose[i][j * 11 + 4],
                    'alpha_max': players_np_ob_considerlose[i][j * 11 + 5],
                    'theta_max': players_np_ob_considerlose[i][j * 11 + 6],
                    'dv': players_np_ob_considerlose[i][j * 11 + 7],
                    'radar': players_np_ob_considerlose[i][j * 11 + 8:j * 11 + 11],
                }
            env_obs['player' + str(i)]['b_info'] = {'b_pos': players_np_ob_considerlose[i][4 * 11:-1],
                                                    'b_visible': players_np_ob_considerlose[i][-1], }
        return env_obs, self.rew, self.ob_done, self.ob_info

    def find_tar(self):
        # 查找函数，目前尚未实现不同距离锁定概率不同
        bp = self.B.get_pos()  # 获取B的位置
        find_num = 0
        for i in list(range(4)):
            ap = self.A[i].get_pos()  # 获取A[i]的位置
            asee = self.A[i].get_see(self.vis)  # 获取A[i]的观测示意线目标点
            disx_temp = (ap[0] - bp[0]) ** 2 + (ap[1] - bp[1]) ** 2 + (ap[2] - bp[2]) ** 2  # 计算A和B的距离
            b_asee = (asee[0] - bp[0]) ** 2 + (asee[1] - bp[1]) ** 2 + (asee[2] - bp[2]) ** 2  # 计算A的正观测点和B的距离
            thetaxcos_temp = (disx_temp + self.vis ** 2 - b_asee) / (
                    2 * (disx_temp ** 0.5) * self.vis)  # 计算A对B的观测偏角，也就是角Asee_A_B的角度
            # 以下限制为边缘测试，在精度不确定的情况下，角度计算可能会溢出，cos范围正常应为[-1,1]
            if thetaxcos_temp > 1:
                thetaxcos_temp = 1
            if thetaxcos_temp < -1:
                thetaxcos_temp = -1
            thetax_temp = math.acos(thetaxcos_temp)  # 计算弧度值
            if (disx_temp <= self.vis ** 2 and thetax_temp <= 15 / 57.3) or (
                    disx_temp <= (self.vis * 2) ** 2 and thetax_temp <= 15 / 57.3 and random.random() <= 0.09):
                # 若找到，则返回A的编号i+1,距离，偏移角度
                find_num = find_num + 1
                self.find = True
                self.find_step = self.step_cum
                self.rew[i] = 1 * max((1000 - self.find_step) / 1000, 0)
                self.ob_info['eprew'] = self.ob_info['eprew'] + self.rew[i]
        return find_num  # 返回找到B的A的数量和奖励

    def hit_tar(self):
        bp = self.B.get_pos()  # 获取B的位置
        for i in list(range(4)):
            ap = self.A[i].get_pos()  # 获取A[i]的位置
            disx_temp1 = (ap[0] - bp[0]) ** 2 + (ap[1] - bp[1]) ** 2
            disx_temp2 = (ap[2] - bp[2]) ** 2  # 计算A和B的距离
            if self.hit[i] is False:
                if disx_temp1 <= self.abdistance[i][0] and disx_temp2 <= self.abdistance[i][1]:
                    if disx_temp1 + disx_temp2 <= 900 and disx_temp2 <= 400:
                        self.hit[i] = True
                        self.A[i].set_live(0)
                        tempstatus = self.A[i].get_status()
                        self.hit_v[i] = tempstatus[1]
                        self.rew[i] = self.rew[i] + 5 * max(((1000 - (self.step_cum - self.find_step)) / 1000), 0)
                    else:
                        self.rew[i] = self.rew[i] + 0.1
                else:
                    self.rew[i] = self.rew[i] - 0.1
                self.ob_info['eprew'] = self.ob_info['eprew'] + self.rew[i]
                self.abdistance[i][0] = disx_temp1
                self.abdistance[i][1] = disx_temp2
        return 0

    def reset(self):
        self.A = []  # 智能体A种群初始化
        for i in list(range(4)):
            # 初始化A
            self.A.append(Agent_A())
        self.B = Agent_B()  # 初始化B
        np_ob0 = []
        self.ob_done = False
        self.hit = [False, False, False, False]
        self.find = False
        self.find_step = 0
        self.abdistance = [[999999999, 999999999], [999999999, 999999999],
                           [999999999, 999999999], [999999999, 999999999]]
        self.hit_num = 0
        self.hit_v = [0, 0, 0, 0]
        self.ob_info = {'eprew': 0, 'eplen': 0, 'pldone': self.hit, 'final_v': self.hit_v}
        for i in list(range(4)):
            # 将A的位置信息、状态信息和预测信息放到信息列表中
            np_ob0 = np_ob0 + self.A[i].get_pos() + self.A[i].get_status() + self.A[i].get_see(self.vis)
        np_ob0 = np_ob0 + self.B.get_pos() + self.B.get_visible()
        np_ob = np.asarray(np_ob0)
        np_ob = np_ob.reshape((48,))
        players_np_ob = self.players_ob(np_ob)
        self.rew = [0, 0, 0, 0]
        self.step_cum = 0
        if RENDER is True:
            self.display_surf.fill((0, 191, 255))
            pygame.display.update()
        env_obs = {'player0': dict(), 'player1': dict(), 'player2': dict(), 'player3': dict(), }
        for i in list(range(4)):
            for j in list(range(4)):
                env_obs['player' + str(i)]['player_' + str(j)] = {
                    'pos': players_np_ob[i][j * 11 + 0:j * 11 + 3],
                    'theta': players_np_ob[i][j * 11 + 3],
                    'v': players_np_ob[i][j * 11 + 4],
                    'alpha_max': players_np_ob[i][j * 11 + 5],
                    'theta_max': players_np_ob[i][j * 11 + 6],
                    'dv': players_np_ob[i][j * 11 + 7],
                    'radar': players_np_ob[i][j * 11 + 8:j * 11 + 11],
                }
            env_obs['player' + str(i)]['b_info'] = {'b_pos': players_np_ob[i][4 * 11:-1],
                                                    'b_visible': players_np_ob[i][-1], }
        return env_obs

    def players_ob(self, raw_np_ob):
        players_np_ob = [raw_np_ob.copy(), raw_np_ob.copy(), raw_np_ob.copy(), raw_np_ob.copy()]
        for i in list(range(4)):
            for j in list(range(4)):
                players_np_ob[i][j * 11 + 0] = players_np_ob[i][j * 11 + 0] - raw_np_ob[i * 11 + 0]
                players_np_ob[i][j * 11 + 1] = players_np_ob[i][j * 11 + 1] - raw_np_ob[i * 11 + 1]
                players_np_ob[i][j * 11 + 2] = players_np_ob[i][j * 11 + 2] - raw_np_ob[i * 11 + 2]
                players_np_ob[i][j * 11 + 8] = players_np_ob[i][j * 11 + 8] - raw_np_ob[i * 11 + 0]
                players_np_ob[i][j * 11 + 9] = players_np_ob[i][j * 11 + 9] - raw_np_ob[i * 11 + 1]
                players_np_ob[i][j * 11 + 10] = players_np_ob[i][j * 11 + 10] - raw_np_ob[i * 11 + 2]
            players_np_ob[i][44] = players_np_ob[i][44] - raw_np_ob[i * 11 + 0]
            players_np_ob[i][45] = players_np_ob[i][45] - raw_np_ob[i * 11 + 1]
            players_np_ob[i][46] = players_np_ob[i][46] - raw_np_ob[i * 11 + 2]
        return players_np_ob

    def players_ob_lose(self, pl_np_ob):
        if self.hit_num != 0:
            for i in list(range(4)):
                for j in list(range(4)):
                    if self.hit[j] is True:
                        for k in list(range(11)):
                            pl_np_ob[i][j * 11 + k] = 0
        return pl_np_ob


def make_env_f4v1():
    return Env()
