import pygame
from pygame.locals import *
from raw_env.env_def import *
from pygame import gfxdraw


class DisplayPlayer(object):

    def __init__(self, display_surf):
        self.screen = display_surf
        self.scr_height = HEIGHT
        self.scr_width = WIDTH
        self.scr_deepheight = DEEPHEIGHT
        self.running = True
        self.vis0 = 1000 / 6000 * WIDTH

    def update(self, raw_obs):
        self.draw_window(raw_obs)

    def draw_window(self, objects):
        # # 可视化
        if not self.running:
            return
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False
            elif event.type == QUIT:
                self.running = False
        if not self.running:
            self.close_window()
            return
        pygame.draw.line(self.screen, (255, 250, 250), (WIDTH, 0), (WIDTH, HEIGHT + DEEPHEIGHT), 1)
        pygame.draw.line(self.screen, (255, 250, 250), (0, HEIGHT), (WIDTH + DEEPHEIGHT, HEIGHT), 1)
        bianjie = (100, 149, 237)
        pygame.draw.circle(self.screen, bianjie, [int(0.5 * self.scr_width),
                                                  int(0.5 * self.scr_height)], int(0.4 * self.scr_height),
                           2)  # 绘制环境范围（3km）

        pygame.draw.line(self.screen, bianjie, (int(0.1 * self.scr_width), int(0.5 * self.scr_height)),
                         (int(0.1 * self.scr_width), HEIGHT + DEEPHEIGHT), 1)
        pygame.draw.line(self.screen, bianjie, (int(0.9 * self.scr_width), int(0.5 * self.scr_height)),
                         (int(0.9 * self.scr_width), HEIGHT + DEEPHEIGHT), 1)

        drawA_color = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
        drawA = []  # A坐标集
        drawA_see = []  # A的正探测点坐标集
        for i in list(range(1, 5)):
            drawA.append([self.norm(objects[i]['A_' + str(i - 1) + '_position'][0]),
                          self.norm(objects[i]['A_' + str(i - 1) + '_position'][1]),
                          self.normdeep(objects[i]['A_' + str(i - 1) + '_position'][2])])
            drawA_see.append([self.norm(objects[i]['A_' + str(i - 1) + '_see'][0]),
                              self.norm(objects[i]['A_' + str(i - 1) + '_see'][1]),
                              self.normdeep(objects[i]['A_' + str(i - 1) + '_see'][2])])
            # 绘制A点

            v_font = pygame.font.Font(None, 20)
            if objects[i]['A_' + str(i - 1) + '_live'][0] == 0 or objects[0] == 10:
                v_color = (219, 112, 147)
                drawA_color[i - 1] = (211, 211, 211)
            else:
                v_color = (255, 255, 255)
            self.draw_one(drawA[i - 1][0], drawA[i - 1][1], drawA[i - 1][2], drawA_color[i - 1])
            self.draw_one_deep(drawA[i - 1][0], drawA[i - 1][1], drawA[i - 1][2], drawA_color[i - 1])
            self.screen.blit(
                v_font.render('A' + str(i - 1) + '_v:' + str(round(objects[i]['A_' + str(i - 1) + '_status'][1], 2)),
                              True, v_color), (int(0.85 * self.scr_width), int((0.03 * i) * self.scr_height)))
            # 绘制正观测示意线
            if objects[i]['A_' + str(i - 1) + '_live'][0] != 0 and objects[0] != 10:
                gfxdraw.pie(self.screen, int(drawA[i - 1][0] * self.scr_width),
                            int((1 - drawA[i - 1][1]) * self.scr_height), int(self.vis0),
                            int((-57.3) * objects[i]['A_' + str(i - 1) + '_status'][0] - 15),
                            int((-57.3) * objects[i]['A_' + str(i - 1) + '_status'][0] + 15), (255, 0, 255))
        # 绘制B点
        if objects[0] != 0:
            drawB = [self.norm(objects[5]['B_position'][0]), self.norm(objects[5]['B_position'][1]),
                     self.normdeep(objects[5]['B_position'][2])]
            self.draw_one_deep(drawB[0], drawB[1], drawB[2], RED, size=6)
            self.draw_one(drawB[0], drawB[1], drawB[2], RED, size=10)
        return self.running

    def norm(self, u):
        # 坐标转换
        return (u + 3000) / 6000

    def normdeep(self, u):
        # 坐标转换
        return (-u) / 240

    def draw_one(self, x, y, d, color_my=(255, 255, 255), shape_type="circle", size=4):
        # 绘制智能体以及智能体文本坐标
        center = int(x * self.scr_width), int((1 - y) * self.scr_height)  # 智能体在可视化窗口中的位置
        if shape_type == "circle":
            # 绘制智能体
            pygame.draw.circle(self.screen, color_my, center, size)
        pos_font = pygame.font.Font(None, 20)  # 字体格式
        self.screen.blit(pos_font.render(
            '(' + str(int(x * 6000 - 3000)) + ',' + str(int(y * 6000 - 3000)) + ',' + str(int(-d * 240)) + ')',
            True, (255, 255, 255)), center)  # 打印坐标

    def draw_one_deep(self, x, y, d, color_my=(255, 255, 255), shape_type="circle", size=3):
        # 绘制智能体以及智能体文本坐标
        center1 = int(x * self.scr_width), int(self.scr_height + d * self.scr_deepheight)
        center2 = int(self.scr_width + d * self.scr_deepheight), int((1 - y) * self.scr_height)
        if shape_type == "circle":
            # 绘制智能体
            pygame.draw.circle(self.screen, color_my, center1, size)

    @staticmethod
    def close_window():
        pygame.display.quit()
        pygame.quit()
