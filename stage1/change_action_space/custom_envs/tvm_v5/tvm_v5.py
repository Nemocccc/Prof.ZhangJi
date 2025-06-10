# from tvm_v3_in import *
from custom_envs.tvm_v5.tvm_v5_in import *
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pygame
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

#参数设置
pygame.init()
FPS = 8
RED = (255, 0, 0)  # 红色
YELLOW = (255, 255, 0)  # 黄色
GREEN = (0, 255, 0)  # 绿色
COLORS=[RED,YELLOW,GREEN]
#width = 800
#length = 800


def circles_collide(x1, y1, x2, y2):
    #等比例扩大
    r=0.5
    # 计算两个圆心之间的距离
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # 判断距离是否小于或等于半径之和
    if distance <= (r*2):
        return True  # 两个圆相撞
    else:
        return False  # 两个圆不相撞



def check_red_blue(red:Red,blue:Blue,decision_step:int)->tuple:
    rew_info={}
    truncated=False
    terminated=False
    rew_info['alive_rew']=0.1

    for monster in red.monster_list:
        if not monster.alive:
            continue
        for bullet in blue.bullet_list:
            if not bullet.alive:
                continue
            mx,my=monster.x,monster.y
            bx,by=bullet.x,bullet.y
            is_collision=circles_collide(mx,my,bx,by)

            if bx<0 or bx>10 or by<0 or by>10:
                bullet.alive=False
                rew_info['hit_fail']=-bullet.hit
                blue.score-=bullet.hit
                continue

            if is_collision:
                monster.hp-=bullet.hit
                bullet.alive=False
                rew_info['hit_succ']=bullet.hit
            
            if monster.hp<=0:
                monster.alive=False
                rew_info['monster_dead']=1
                blue.score+=1
                break
    
    for monster in red.monster_list:
        if not monster.alive:
            continue
        if monster.y <=0:
            monster.alive=False
            rew_info['monster_attack']=-2
            blue.score-=1
            blue.hp-=1
        if blue.hp<=0:
            truncated=True
            rew_info['blue_dead']=-5
            break
    if decision_step+1>=200:
        rew_info['red_dead']=5
        terminated=True
    return rew_info,terminated,truncated

class TVM(gym.Env):
    #将render_fps 改为用FPS统一管理 主函数中控制画面帧率也会用到
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None) -> None:
        super().__init__()

        self.observation_space=gym.spaces.Box(low=0,high=np.inf,shape=(14,),dtype=np.float32)
        self.action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(17,), dtype=np.float32)

        self.blue=Blue()
        self.red=Red()

        self.info={'eplen':0,'eprew':0}

        self.decision_step=0

        self.fig, self.ax = plt.subplots(figsize=(5, 5))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        self.obs_num_upper=4
        
        self.obs=None

        #设置窗口大小 并设置时钟用于控制帧率
        self.screen = pygame.display.set_mode((800, 800))
        self.clock = pygame.time.Clock()

        # 设置字体
        self.font = pygame.font.SysFont("Arial", 24)
    
    def reset(self,seed=42,options=None):
        np.random.seed(seed)
        random.seed(seed)
        self.info={'eplen':0,'eprew':0}
        info={}
        self.decision_step=0
        self.red.reset()
        self.blue.reset()

        if options is not None:
            self.red.set_monster_num(options['monster_num'])

        red_states=self.red.get_states()
        blue_states=self.blue.get_states()
        
        real_red_states=red_states
        real_blue_states=blue_states

        all_states={"red_states":real_red_states,"blue_states":real_blue_states}

        info={}
        return all_states,info

    def step(self,action):
        info={}
        self.decision_step+=1
        
        rew=0
        terminated=False
        truncated=False
        self.blue.update_states(self.decision_step,action)
        self.red.update_states(self.decision_step)

        rew_info,terminated,truncated=check_red_blue(self.red,self.blue,self.decision_step)

        red_states=self.red.get_states()
        blue_states=self.blue.get_states()

        real_red_states=red_states
        real_blue_states=blue_states

        all_states={"red_states":real_red_states,"blue_states":real_blue_states}

        for key,item in rew_info.items():
            rew+=item

        return all_states,rew,terminated,truncated,info
        
    def update_obs(self,obs):
        # 确保 self.obs 被正确赋值
        self.obs = obs  # 而不是 self.obs = None
        
    def render(self):
        obs=self.obs

        if obs is not None:
            select_monster=[[obs[0],obs[1]],[obs[4],obs[5]],[obs[8],obs[9]]]
        else:
            select_monster=[[0,0],[0,0],[0,0]]
        
        self.screen.fill('white')


        #画背景
        img = pygame.image.load('custom_envs/tvm_v3/IMG/back_ground_2.png').convert_alpha()
        
        Rect = img.get_rect()
        scale_factor = 2.8
        back_ground_img = pygame.transform.scale(img, (int(Rect.width * scale_factor), int(Rect.height * scale_factor)))
        self.screen.blit(back_ground_img, (0,0))

        #画基地
        img = pygame.image.load('custom_envs/tvm_v3/IMG/home_light.png').convert_alpha()
        Rect = img.get_rect()
        scale_factor = 0.3
        blue_img = pygame.transform.scale(img, (int(Rect.width * scale_factor), int(Rect.height * scale_factor)))
        self.screen.blit(blue_img, (int(self.blue.x*80-35.5), int(800-self.blue.y*80-72)))

        # 基地上方显示HP
        blue_hp_text = self.font.render(f"HP: {self.blue.hp:.2f}", True, (0, 0, 0))
        self.screen.blit(blue_hp_text, (int(self.blue.x*80-35.5), int(800-self.blue.y*80-72) - 30)) 
        
        # 画子弹
        img = pygame.image.load('custom_envs/tvm_v3/IMG/bullet_light.png').convert_alpha()
        Rect = img.get_rect()
        scale_factor = 0.25
        bullet_img = pygame.transform.scale(img, (int(Rect.width * scale_factor), int(Rect.height * scale_factor)))

        for bullet in self.blue.bullet_list:
            if not bullet.alive:
                continue
            # alpha_value = max(80, 255 * (bullet.v-0.5)/1.5)  # 最大透明度为255，最小为50
            alpha_value=255
            bullet_img.set_alpha(alpha_value)  # 设置透明度
            self.screen.blit(bullet_img, (int(bullet.x*80-35.5), int(800-bullet.y*80-72)))


            # 显示子弹的伤害
            bullet_damage_text = self.font.render(f" {bullet.hit:.2f}", True, (255, 0, 0))
            self.screen.blit(bullet_damage_text, (int(bullet.x*80-35.5), int(800-bullet.y*80-72) - 20))  # 子弹上方显示伤害数值

        # 画怪物
        img = pygame.image.load('custom_envs/tvm_v3/IMG/monster_light.png').convert_alpha()
        Rect = img.get_rect()
        scale_factor = 0.06
        monster_img = pygame.transform.scale(img, (int(Rect.width * scale_factor), int(Rect.height * scale_factor)))

        for monster in self.red.monster_list:
            if not monster.alive:
                continue
            # alpha_value = max(80, 255 * (monster.v*80 / 80))  # 最大透明度为255，最小为80
            alpha_value=255
            monster_img.set_alpha(alpha_value)  # 设置透明度
            self.screen.blit(monster_img, (int(monster.x*80), int(800-monster.y*80)))

            # 绘制血条
            bar_width = 50  # 血条宽度
            bar_height = 6  # 血条高度
            bar_x = int(monster.x*80) - bar_width + 60 # 血条X坐标，居中显示
            bar_y = int(800-monster.y*80) -10 # 血条Y坐标，放置在怪物上方

            # 绘制血条背景（灰色）
            pygame.draw.rect(self.screen, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))

            # 绘制当前HP（绿色到红色的渐变）
            hp_percentage = max(0, monster.hp / 1)  # 计算血条的比例
            #用RGB通道表示血条颜色 
            hp_color = (255 - int(255 * hp_percentage), int(255 * hp_percentage), 0)  # 红到绿渐变
            pygame.draw.rect(self.screen, hp_color, (bar_x, bar_y, bar_width * hp_percentage, bar_height))

            # 显示怪物的HP（保留2位小数）
            monster_hp_text = self.font.render(f"v={monster.v:.2f}", True, (0, 0, 0))
            self.screen.blit(monster_hp_text, (bar_x + bar_width // 2 - monster_hp_text.get_width() // 2, bar_y - 20))  # 显示在血条上方

            # 显示得分 此处有点问题（得分一直为负） 可能是因为v0和v3的得分计算不一致导致

            # score_text = self.font.render(f'Score: {self.blue.score:.2f}', True, (255, 0, 0))  # 文字内容，是否抗锯齿，颜色
            # self.screen.blit(score_text, (10, 10))  # 在屏幕的左上角绘制得分
       
        # for color,coord in zip(COLORS,select_monster):
        #     x, y = coord
        #     pygame.draw.circle(self.screen, color, (int(x * 80+40), int(800 - y * 80+20)), 50, width=5)  # 绘制半径为 50 的红色圆圈，宽度为 5

        pygame.display.update()
        self.clock.tick(FPS)
        return self.get_screen_image()

    def get_screen_image(self):
        screen_data = pygame.surfarray.array3d(self.screen)
        screen_data = np.flip(screen_data, axis=1)  # 水平翻转
        image = Image.fromarray(screen_data)
        rotated_image = image.rotate(90, expand=True)  # 逆时针旋转90度，expand=True确保图像不会被裁剪
        return rotated_image
        

    def close(self):
        
        pass
   
if __name__ == "__main__":
    env=TVM()
    done=False
    obs=env.reset(seed=35)

    #设置窗口大小 并设置时钟用于控制帧率
    self.screen = pygame.display.set_mode((800, 800))
    clock = pygame.time.Clock()

    while not done:
        action=env.action_space.sample()
        obs,reward,terminated,truncated,info=env.step(action)
        done=terminated or truncated
        env.render()

        #延时 从而控制帧率 
        #也就是控制for循环的速度 但同时也会控制智能体采取动作的速度
        