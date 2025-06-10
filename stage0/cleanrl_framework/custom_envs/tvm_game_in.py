import numpy as np
import random
import math

class Bullet:
    def __init__(self,x:float,y:float,theta:float,v:float,hit:float) -> None:
        self.x=x   # 固定位置
        self.y=y    # 根据速度
        self.theta=theta
        self.v=v    # 固定速度
        self.hit=hit
        self.alive=True
        self.step=0

    def update_states(self)->None:
        tempy=self.y+math.cos(self.theta)*self.v
        tempx=self.x+math.sin(self.theta)*self.v
        self.y=tempy
        self.x=tempx
        self.step+=1

    def get_states(self)->list:
        return [self.x,self.y,self.v,self.theta]

class Blue:
    def __init__(self,x=5.0,y=0.0) -> None:
        self.hp=3
        self.score=0

        # 炮塔的位置
        self.x=x
        self.y=y

        self.available_bullet=3
        self.bullet_list=[]
    
    def reset(self,x=5.0,y=0.0) -> None:
        self.hp=3
        self.score=0
        # 炮塔的位置
        self.x=x
        self.y=y

        self.available_bullet=3
        self.bullet_list=[]
    
    
    def update_states(self,decision_step:int,action_list:list)->None:
        # 根据action_list 决定需要发射多少子弹，每个子弹分配多少伤害，每个子弹的速度和方向
        shoot_num=np.argmax(action_list[:4])
        hit_set=action_list[4:7]
        theta_list=[action_list[i]*np.pi for i in range(7,13,2)] # [-1,1] 映射到 [-pi,pi]
        v_list=[action_list[i]*0.75+1.25 for i in range(8,13,2)] #[-1,1] 映射到[0,2]

        to_remove=[]
        for i,bullet in enumerate(self.bullet_list):
            if not bullet.alive:
                to_remove.append(bullet)
                continue
            bullet.update_states()

        # 移除已经消失的对象
        for item in to_remove:
            self.bullet_list.remove(item)

        # 1个step补充一个子弹
        if decision_step%1==0:
            self.available_bullet=min(self.available_bullet+1,3)

        if shoot_num>self.available_bullet:
            # shoot_num=self.available_bullet    
            return 

        if shoot_num==0:
            return


        scores_array=hit_set[:shoot_num]
        softmax_scores = np.exp(scores_array) / np.sum(np.exp(scores_array)) #保证最后伤害总和为1
        
        for i in range(shoot_num):
            self.available_bullet-=1
            self.bullet_list.append(Bullet(self.x,self.y,theta_list[i],v_list[i],softmax_scores[i]))
        

    def get_states(self)->list:
        bullet_states=[0]*4
        bullet_states[self.available_bullet]=1
        all_states=bullet_states+[self.x,self.y,self.hp]
        return all_states

class Monster:
    def __init__(self) -> None:
        self.x=0   
        self.y=0
        self.v=0    # 固定速度
        self.hp=0   # 0-1之间随机
        self.alive=False #存活标记
        self.step=0
    
    def reset(self) -> None:
        self.x=0   
        self.y=0
        self.v=0    # 固定速度
        self.hp=0   # 0-1之间随机
        self.alive=False #存活标记
        self.step=0

    def set_states(self)->None:
        self.x=random.uniform(0,10)    
        self.y=10
        self.v=-0.5    # 固定速度
        self.hp=random.uniform(0.1, 0.5)   # 0-1之间随机
        self.alive=True #存活标记
        self.step=0

    def update_states(self)->None:
        tempy=self.y+self.v
        self.y=tempy
        self.step+=1


    def get_states(self)->list:
        inf=10+1
        if not self.alive:
            return [inf,inf,-inf,-inf]
        
        return [self.x,self.y,self.v,self.hp]
    

class Red:
    def __init__(self) -> None:
        self.monster_list=[]
        self.monster_num=6
        for i in range(self.monster_num):
            self.monster_list.append(Monster())   
    
    def reset(self) -> None:
        
        for i in range(self.monster_num):
            self.monster_list[i].reset()   
    
    
    def update_states(self,decision_step)->None:
        
        # 红方每4个时间步生成1到3个怪物
        if decision_step%4==0:
            monster_num=random.randint(1, 3)
        else:
            monster_num=0
                    
        for monster in self.monster_list:
            if not monster.alive:
                if monster_num>0:
                    monster.set_states()
                    monster_num-=1
            else:
                monster.update_states()

    def get_states(self)->list:
        states_list=[]
        for monster in self.monster_list:
            states_list.append(monster.get_states())
            
        return states_list