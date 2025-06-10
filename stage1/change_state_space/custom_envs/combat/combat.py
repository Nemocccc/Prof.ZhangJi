import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import defaultdict
import pygame


class VisualMABattleEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array',None], 'render_fps': 2}

    def __init__(self,
                 n_agents=3,
                 grid_size=10,
                 max_steps=500,
                 render_mode=None):

        super().__init__()

        # 环境参数
        self.n_agents=n_agents
        self.red_agents = n_agents//2
        self.blue_agents = n_agents
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Pygame配置
        self.cell_size = 40
        self.window_size = self.grid_size * self.cell_size
        self.colors = {
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'background': (255, 255, 255),
            'grid': (200, 200, 200)
        }

        self.total_reward=0
        # 初始化Pygame
        if self.render_mode == 'human':
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("MA Combat Environment")
            self.clock = pygame.time.Clock()  # 添加时钟对象

        # 动作空间定义
        self.action_space =  spaces.Discrete(5)

        # 观察空间定义：每个蓝方智能体的观察数据是 61 维
        # self.observation_space = spaces.Box(
        #     low=0, high=100, shape=(61 * blue_agents,), dtype=np.float32
        # )
        self.observation_space = spaces.Box(low=0, high=100, shape=(52,), dtype=np.float32)


        # 实体存储
        self.entities = {}
        self.agent_ids = []

        # 状态缓存
        self.current_step = 0
        self.comm_states = {}
        self._init_positions()

    def _init_positions(self):
        """初始化实体位置"""
        positions = set()

        # 生成红方位置
        for i in range(self.red_agents):
            while True:
                pos = (np.random.randint(0, self.grid_size),
                       np.random.randint(0, self.grid_size))
                if pos not in positions:


                    x, y = pos
                    
                    # 周围环境感知：3x3网格，3个值（红方、蓝方、空地）
                    surroundings = np.zeros((3, 3, 2), dtype=np.float32)
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            px = x + dx
                            py = y + dy
                            if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
                                for e in self.entities.values():
                                    if e['pos'] == (px, py):
                                        if e['team'] == 'red':
                                            surroundings[dx + 1, dy + 1, 0] = 1
                                        elif e['team'] == 'blue':
                                            surroundings[dx + 1, dy + 1, 1] = 1
                                        break


                    self.entities[f"red_{i}"] = {
                        'team': 'red',
                        'pos': pos,
                        'hp': 3,
                        'cooldown': 0,

                    }
                    positions.add(pos)
                    break

        # 生成蓝方位置
        for i in range(self.blue_agents):
            while True:
                pos = (np.random.randint(0, self.grid_size),
                       np.random.randint(0, self.grid_size))
                if pos not in positions:

                    x, y = pos
                    
                    # 周围环境感知：3x3网格，3个值（红方、蓝方、空地）
                    surroundings = np.zeros((3, 3, 2), dtype=np.float32)
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            px = x + dx
                            py = y + dy
                            if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
                                for e in self.entities.values():
                                    if e['pos'] == (px, py):
                                        if e['team'] == 'red':
                                            surroundings[dx + 1, dy + 1, 0] = 1
                                        elif e['team'] == 'blue':
                                            surroundings[dx + 1, dy + 1, 1] = 1
                                        break
                                # else:
                                #     surroundings[dx + 1, dy + 1, 2] = 1

                    self.entities[f"blue_{i}"] = {
                        'team': 'blue',
                        'pos': pos,
                        'hp': 3,
                        'cooldown': 0,

                    }
                    positions.add(pos)

                    break
        
        for i in range(self.blue_agents):
            agent_id = f"blue_{i}"
            entity = self.entities.get(agent_id)

            if entity and entity['hp'] > 0:  # 如果智能体存活
                x, y = entity['pos']

                # 周围环境感知：3x3网格，3个值（红方、蓝方、空地）
                surroundings = np.zeros((3, 3, 2), dtype=np.float32)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        px = x + dx
                        py = y + dy
                        if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
                            for e in self.entities.values():
                                if e['pos'] == (px, py):
                                    if e['team'] == 'red':
                                        surroundings[dx + 1, dy + 1, 0] = 1
                                    elif e['team'] == 'blue':
                                        surroundings[dx + 1, dy + 1, 1] = 1 
                                    break
                            # else:
                            #     surroundings[dx + 1, dy + 1, 2] = 1


    def _get_obs(self):
        """生成共享的观察数据列表"""
        observations = []

        for i in range(self.blue_agents):
            agent_id = f"blue_{i}"
            entity = self.entities.get(agent_id)

            if entity and entity['hp'] > 0:  # 如果智能体存活
                x, y = entity['pos']
                self_state = np.array([x, y, entity['hp'], entity['cooldown']], dtype=np.float32)

                # 周围环境感知：3x3网格，3个值（红方、蓝方、空地）
                surroundings = np.zeros((3, 3, 2), dtype=np.float32)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        px = x + dx
                        py = y + dy
                        if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
                            for e in self.entities.values():
                                if e['pos'] == (px, py):
                                    if e['team'] == 'red':
                                        surroundings[dx + 1, dy + 1, 0] = 1
                                    elif e['team'] == 'blue':
                                        surroundings[dx + 1, dy + 1, 1] = 1
                                    break
                            # else:
                            #     surroundings[dx + 1, dy + 1, 2] = 1

                # 消息：最多10条消息，每条消息包含3个值（位置，生命值）
                messages = np.zeros((10, 3), dtype=np.float32)
                msg_count = 0
                for sender_id, msg in self.comm_states.items():
                    # 只处理存活且是蓝方的智能体
                    if self.entities.get(sender_id, {}).get('hp', 0) > 0:
                        if msg_count < 10:

                            messages[msg_count] = np.array(msg)
                            msg_count += 1

            else:
                # 死亡的智能体使用零向量作为占位符
                self_state = np.zeros(4, dtype=np.float32)
                surroundings = np.zeros((3, 3, 2), dtype=np.float32)
                messages = np.zeros((10, 3), dtype=np.float32)

            # 将每个智能体的观测（61维）添加到列表中
            
            observations.append(list(self_state.flatten())+list(surroundings.flatten())+list(messages.flatten()))
        return np.array(observations, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_positions()
        self.current_step = 0
        self.total_reward=0
        # 更新通信状态：保存每个智能体的位置信息和生命值
        self.comm_states = {
            agent_id: [entity['pos'][0],entity['pos'][1],entity['hp']]
            for agent_id, entity in self.entities.items()
        }

        # 返回共享的观测空间（包含所有蓝方智能体的观察）
        return self._get_obs(), {}

    def step(self, raw_actions):
        # 定义动作映射：将单维度动作映射到二维动作
        action_mapping = {
            0: 0,  # 上 + 不攻击
            1: 1,  # 上 + 攻击
            2: 2,  # 下 + 不攻击
            3: 3,  # 下 + 攻击
            4: 4,  # 左 + 不攻击
        }
        
        actions={f"blue_{i}":raw_actions[i] for i in range(self.n_agents)}

        # 过滤掉已经被摧毁的智能体的动作
        valid_actions = {
            agent_id: action_mapping[action]
            for agent_id, action in actions.items()
            if agent_id in self.entities
        }

        # 更新通信状态：保存每个智能体的位置信息和生命值
    
        self.comm_states = {
            agent_id: [entity['pos'][0],entity['pos'][1],entity['hp']]
            for agent_id, entity in self.entities.items()
        }

        # 处理红方智能体的随机移动
        positions=[]
        for agent_id, entity in self.entities.items():
            positions.append(entity['pos'])

        attack_list = []

        tx,ty=0,0
        for agent_id, entity in self.entities.items():
            if agent_id.startswith('blue'):
                tx,ty=entity['pos']
                break

        rewards = {f"blue_{i}":0.0 for i in range(self.blue_agents)}

        for agent_id, entity in self.entities.items():
            if agent_id.startswith('red'):
                nx, ny = entity['pos']
                # action =   # 随机选择移动方向
                move=np.random.randint(0, 5)
                if entity['cooldown']<=0 and self._check_attack_target(agent_id):
                    attack_list.append(agent_id)
                    # print(agent_id)
                elif move == 0:  # 上
                    ny = min(ny + 1, self.grid_size - 1)
                elif move == 1:  # 下
                    ny = max(ny - 1, 0)
                elif move == 2:  # 左
                    nx = max(nx - 1, 0)
                elif move == 3:  # 右
                    nx = min(nx + 1, self.grid_size - 1)

                if (nx,ny) not in positions:
                    entity['pos'] = (nx, ny)
                    positions.append((nx,ny))

        # 处理蓝方智能体的移动
        for agent_id, action in valid_actions.items():
            if agent_id.startswith('blue'):    
                entity = self.entities[agent_id]
                nx, ny = entity['pos']
                move = action  # 随机选择移动方向
                if move == 0:  # 上
                    ny = min(ny + 1, self.grid_size - 1)
                elif move == 1:  # 下
                    ny = max(ny - 1, 0)
                elif move == 2:  # 左
                    nx = max(nx - 1, 0)
                elif move == 3:  # 右
                    nx = min(nx + 1, self.grid_size - 1)
                else:
                    # rewards[agent_id]-=0.1
                    if entity['cooldown']>0:
                        attack_list.append(agent_id)
                            
                if (nx,ny) not in positions:
                    entity['pos'] = (nx, ny)
                    positions.append((nx,ny))
                else:
                    rewards[agent_id]+=-0.01

        # 处理攻击


        # 更新冷却时间
        for agent_id in self.entities.keys():
            if self.entities[agent_id]['cooldown'] > 0:
                self.entities[agent_id]['cooldown'] -= 1

        # 执行攻击
        for attacker_id in attack_list.copy():  # 使用副本遍历以避免迭代时修改原列表
            # 检查攻击者是否已被销毁
            if attacker_id not in self.entities:
                continue
            attacker = self.entities[attacker_id]
            # 检查攻击者是否处于冷却状态
            if attacker['cooldown'] > 0:
                continue
            # 获取攻击目标（确保目标存在）
            targets = [
                target_id
                for target_id in self._get_attack_targets(attacker_id)
                if target_id in self.entities  # 只保留未被销毁的目标
            ]
            for target_id in targets:
                target = self.entities[target_id]
                target['hp'] -= 1
                if attacker_id in rewards.keys():
                    rewards[attacker_id] += 1
                if target_id in rewards.keys():
                    rewards[target_id] -= 0.5
                # 检查目标是否死亡
                if target['hp'] <= 0:
                    for key,value in rewards.items():
                        rewards[key]+= 2/self.blue_agents
                    # if attacker_id in rewards.keys():
                    #     rewards[attacker_id] += 1
                    # if target_id in rewards.keys():
                    #     rewards[target_id] -= 1
                    del self.entities[target_id]
            # 更新攻击者冷却时间
            attacker['cooldown'] = 1


        # 检查终止条件
        terminated = False
        red_alive = any(e['team'] == 'red' for e in self.entities.values())
        blue_alive = any(e['team'] == 'blue' for e in self.entities.values())

        # if not red_alive:
        #     terminated = True
        #     for key,value in rewards.items():
        #         rewards[key]+=10

        # elif not blue_alive:
        #     terminated = True
        #     for key,value in rewards.items():
        #         rewards[key]-=10
                
        if not red_alive:
            terminated = True
        elif not blue_alive:
            terminated = True

        # te_list=[]
        # for i in range(self.n_agents):

        #     if f"blue_{i}" in self.entities.keys():
        #         te_list.append(False)
        #     else:
        #         te_list.append(True)
        

        # 生成共享的观察空间（蓝方所有智能体的观察）
        obs = self._get_obs()
        self.current_step += 1

        red_alive_num = sum(e['team'] == 'red' for e in self.entities.values())
        blue_alive_num = sum(e['team'] == 'blue' for e in self.entities.values())

        for key,value in rewards.items():
            rewards[key]+=-0.01

        te_list=[False]*self.n_agents
        if self.current_step>self.max_steps or terminated:
            for key,value in rewards.items():
                rewards[key]+=-red_alive_num/self.blue_agents
            
            for i in range(self.n_agents):
                    te_list[i]=True           

        rewards_list=[]
        for key,value in rewards.items():
            rewards_list.append(value)
        # print(te_list)
        self.total_reward+=sum(rewards_list)
        return obs, rewards_list, np.array(te_list), np.array([False]*self.n_agents), {}

    def _check_attack_target(self, attacker_id):
        """检查攻击范围内是否有目标"""
        attacker = self.entities[attacker_id]
        x, y = attacker['pos']

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                px = x + dx
                py = y + dy
                if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
                    for target_id, target in self.entities.items():
                        if target['pos'] == (px, py) and target['team'] != attacker['team']:
                            return True
        return False

    def _get_attack_targets(self, attacker_id):
        """获取攻击范围内的所有目标"""
        attacker = self.entities[attacker_id]
        x, y = attacker['pos']
        targets = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                px = x + dx
                py = y + dy
                if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
                    for target_id, target in self.entities.items():
                        if (target['pos'] == (px, py) and
                                (target['team'] != attacker['team']) and
                                (target_id != attacker_id)):
                            targets.append(target_id)
                            break
        return targets

    def render(self):
        if self.render_mode is None:
            return

        # 创建画布
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(self.colors['background'])

        # 绘制网格线
        for x in range(self.grid_size):
            pygame.draw.line(
                canvas,
                self.colors['grid'],
                (0, x * self.cell_size),
                (self.window_size, x * self.cell_size),
                width=1
            )
            pygame.draw.line(
                canvas,
                self.colors['grid'],
                (x * self.cell_size, 0),
                (x * self.cell_size, self.window_size),
                width=1
            )

        # 绘制实体
        for agent_id, entity in self.entities.items():
            x, y = entity['pos']
            color = self.colors['red'] if entity['team'] == 'red' else self.colors['blue']

            # 计算屏幕位置
            screen_x = x * self.cell_size + self.cell_size // 2
            screen_y = (self.grid_size - 1 - y) * self.cell_size + self.cell_size // 2

            # 绘制主体
            pygame.draw.circle(
                canvas,
                color,
                (screen_x, screen_y),
                self.cell_size // 3
            )

            # 绘制生命值
            pygame.font.init()
            font = pygame.font.Font(None, 24)
            text = font.render(str(entity['hp']), True, (0, 0, 0))
            canvas.blit(text, (screen_x - 10, screen_y - 10))

            # 绘制通信状态
            if not self.comm_states[agent_id]:
                pygame.draw.line(
                    canvas,
                    (255, 0, 0),
                    (screen_x - 10, screen_y - 10),
                    (screen_x + 10, screen_y + 10),
                    width=3
                )

        # 更新显示
        if self.render_mode == 'human':
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # 控制帧率
            self.clock.tick(self.metadata['render_fps'])  # 限制帧率

            # 处理关闭事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
        else:
            return np.array(np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)),
                axes=(1, 0, 2)
            ))

    def close(self):
        if hasattr(self, 'window') and self.window is not None:
            pygame.display.quit()
            pygame.quit()


# 使用示例
if __name__ == "__main__":
    env = VisualMABattleEnv(
        n_agents=5,
        grid_size=10,
        render_mode='rgb_array'
    )

    obs, _ = env.reset()
    done = False

    n_agents=5
    while not done:
        actions=[]
        for _ in range(n_agents):
            actions.append(env.action_space.sample())
        obs, rewards, terminated, truncated, _ = env.step(actions)
        print(rewards,obs.shape)
        done = terminated[0] or truncated[0]
        image=env.render()

        # print(image)
    env.close()
