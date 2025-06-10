from custom_envs.re_tvm_game_in import *
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

def circles_collide(x1, y1, x2, y2):
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
    sys_info=""
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
                sys_info=sys_info+"子弹出界，"
                continue

            if is_collision:
                monster.hp-=bullet.hit
                bullet.alive=False
                rew_info['hit_succ']=bullet.hit
                sys_info=sys_info+f"子弹击中红方单位，造成伤害值{bullet.hit}，"
            
            if monster.hp<=0:
                monster.alive=False
                rew_info['monster_dead']=1
                blue.score+=1
                sys_info=sys_info+"红方单位死亡，"
                break
    
    for monster in red.monster_list:
        if not monster.alive:
            continue
        if monster.y<=0:
            monster.alive=False
            rew_info['monster_attack']=-2
            blue.hp-=1
            sys_info=sys_info+f"红方单位成功入侵蓝方大本营一次，蓝方当前血量 {blue.hp}，"
        if blue.hp<=0:
            truncated=True
            rew_info['blue_dead']=-5
            sys_info=sys_info+"蓝方当前血量为 0，红方进攻成功，红方取得胜利，"
            break
    if decision_step+1>=200:
        rew_info['red_dead']=5
        sys_info=sys_info+f"蓝方防御成功，蓝方取得胜利，对局蓝方击毁 {blue.score} 个敌方单位，得分 {blue.score}"
        terminated=True
    return rew_info,terminated,truncated,sys_info

class TVM(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None) -> None:
        super().__init__()

        self.observation_space=gym.spaces.Box(low=-11,high=11,shape=(23,),dtype=np.float32)
        self.action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(13,), dtype=np.float32)

        self.blue=Blue()
        self.red=Red()

        self.decision_step=0

        self.fig, self.ax = plt.subplots(figsize=(5, 5))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.info={}
        self.obs_num_upper=4
        
    
    def reset(self,seed=42,options=None):
        np.random.seed(seed)
        random.seed(seed)
        
        self.decision_step=0
        self.red.reset()
        self.blue.reset()
        red_states,red_info=self.red.get_states()
        blue_states,blue_info=self.blue.get_states()
        
        real_red_states=[]
        red_states=sorted(red_states,key=lambda x: x[1])[:self.obs_num_upper]
        for rs in red_states:

            real_red_states.extend(rs)
        real_blue_states=blue_states

        all_states=np.array(real_red_states+real_blue_states)

        self.info={"sys_info":"","red_info":red_info,"blue_info":blue_info,"eplen":0,"eprew":0}
        
        return all_states,self.info

    def step(self,action):
        self.decision_step+=1
        info={}
        rew=0
        terminated=False
        truncated=False
        self.blue.update_states(self.decision_step,action)
        self.red.update_states(self.decision_step)

        rew_info,terminated,truncated,sys_info=check_red_blue(self.red,self.blue,self.decision_step)

        red_states,red_info=self.red.get_states()
        blue_states,blue_info=self.blue.get_states()

        real_red_states=[]
        red_states=sorted(red_states,key=lambda x: x[1])[:self.obs_num_upper]
        for rs in red_states:
            real_red_states.extend(rs)
        real_blue_states=blue_states

        all_states=np.array(real_red_states+real_blue_states)

        for key,item in rew_info.items():
            rew+=item

        self.info['sys_info']=sys_info
        self.info['red_info']=red_info
        self.info['blue_info']=blue_info
        self.info['eplen']+=1
        self.info['eprew']+=rew

        # print("rew=",rew)
        return all_states,rew,terminated,truncated,self.info
    

    def render(self):
        # 创建图和轴，调整图形大小
        self.ax.cla()  # 清除之前的图形

        # 设置环境大小
        env_size = 10
        # 设置坐标轴的范围
        self.ax.set_xlim(0, env_size)
        self.ax.set_ylim(0, env_size)

        # 设置坐标轴的比例相同，以确保表示的物体不会因为坐标轴的拉伸而变形
        self.ax.set_aspect('equal')

        # 绘制边界方框
        rect = patches.Rectangle((0, 0), env_size, env_size, linewidth=1, edgecolor='r', facecolor='none')
        self.ax.add_patch(rect)

        # 绘制炮塔（蓝色三角形）
        tower_center = (self.blue.x, self.blue.y)
        self.ax.scatter(tower_center[0], tower_center[1], color='blue', marker='s', s=200)
        # 添加炮塔血量标识
        self.ax.text(tower_center[0] + 0.1, tower_center[1] + 0.1, f'HP: {self.blue.hp:.2f}', color='blue', fontsize=12)

        # 绘制子弹（蓝色三角形）
        for bullet in self.blue.bullet_list:
            if not bullet.alive:
                continue
            bullet_center = (bullet.x, bullet.y)
            self.ax.scatter(bullet_center[0], bullet_center[1], color='blue', marker='*', s=100)
            # 添加子弹伤害值标识
            self.ax.text(bullet_center[0], bullet_center[1], f'{bullet.hit:.2f}', color='blue', fontsize=12)

        # 绘制怪物（红色圆形）
        for monster in self.red.monster_list:
            if not monster.alive:
                continue
            monster_center = (monster.x, monster.y)
            self.ax.scatter(monster_center[0], monster_center[1], color='red', marker='o', s=100)
            # 添加怪物血量标识
            self.ax.text(monster_center[0], monster_center[1], f'{monster.hp:.2f}', color='red', fontsize=12)

         # 在方框外加一个显示blue.score的标签
        self.ax.text(-0.1, 1, f'Score: {self.blue.score:.2f}', color='blue', fontsize=10, transform=self.ax.transAxes)

        # 关闭坐标轴的显示
        self.ax.axis('off')

        if self.render_mode == "rgb_array":
            # 创建一个用于保存图形的画布
            canvas = FigureCanvasAgg(self.fig)
            # 绘制图形
            canvas.draw()
            # 将图形转换为RGB数组
            buf = canvas.buffer_rgba()
            # 返回RGB数组
            return np.uint8(buf)
        else:
            # 显示图形
            plt.draw()
            plt.pause(0.1)  # 暂停一段时间，以便图形更新

    def close(self):
        pass
    
if __name__ == "__main__":
    env=TVM()
    done=False
    obs=env.reset(seed=35)

    while not done:
        action=env.action_space.sample()
        obs,reward,terminated,truncated,info=env.step(action)
        done=terminated or truncated
        env.render()