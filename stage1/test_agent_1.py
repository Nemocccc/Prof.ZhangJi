import json
import custom_envs
import gymnasium as gym
from train_agent_tvm import *
from PIL import Image
import os
import shutil
from custom_envs.tvm_v3.tvm_v3 import TVM

def clear_directory(directory, delete_subfolders=False):
    """
    清除指定目录下的所有文件和（可选）子文件夹。

    :param directory: 要清除的目录路径
    :param delete_subfolders: 是否删除子文件夹，默认为False
    """
    # 检查目录是否存在
    if not os.path.exists(directory):
        print("指定的目录不存在")
        return

    # 遍历目录中的所有文件和文件夹
    for filename in os.listdir(directory):
        # 构造完整的文件路径
        file_path = os.path.join(directory, filename)
        
        # 检查这是一个文件还是文件夹
        if os.path.isfile(file_path) or os.path.islink(file_path):
            # 删除文件或链接
            os.unlink(file_path)
            print(f"已删除文件：{file_path}")
        elif os.path.isdir(file_path):
            if delete_subfolders:
                # 删除文件夹及其内容
                shutil.rmtree(file_path)
                print(f"已删除文件夹及其内容：{file_path}")
            else:
                print(f"子文件夹未被删除：{file_path}")

class Agent(nn.Module):
    def __init__(self, envs=None):
        super().__init__()        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(19, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(19, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 17), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, 17))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def get_action(self, x):
        action_mean = self.actor_mean(x)
        return action_mean



def obs_pipeline(env,raw_obs,encode_mode=0):

    red_states=raw_obs['red_states']
    blue_states=raw_obs['blue_states']

    monster_state_list=red_states['monster_state_list']

    bullet_num=blue_states['bullet_num']
    tower=blue_states['tower']

    obs=[]
    if encode_mode==0:

        monster_state_list=sorted(monster_state_list,key=lambda x: x[1])[:3]
        env.update_obs(np.array(monster_state_list).flatten())
        monster_state_list=sorted(monster_state_list,key=lambda x: x[0])
        for m_state in monster_state_list:
            obs.extend(m_state)

        
    obs.extend(bullet_num)
    obs.extend(tower)            

    return np.array(obs)


if __name__ == "__main__":

    # TRY NOT TO MODIFY: seeding
    args = tyro.cli(Args)
    # args.capture_video=False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env =gym.make("custom_envs/TVM-v4", render_mode="rgb_array")
    env = gym.wrappers.ClipAction(env)
    
    agent = Agent(env).to(device)

    record=False # 是否进行对局信息记录

    monster_num_list=[3]
    train_seed_list=[42]
    
    # 需要自己训练tvm-v1环境下的智能体
    model_path="./modified_model.pth"

    r_data={}
    for monster_num in monster_num_list:
        for agent_type in [0]:


            agent.load_state_dict(torch.load(model_path))
            succ_cnt=0
            max_cnt=100
            for j in range(max_cnt):
                done=False
                options={'monster_num':monster_num}
                raw_obs,info=env.reset(seed=j,options=options)

                while not done:
                    # obs=obs_pipeline(env,raw_obs,agent_type)
                    obs=raw_obs
                    obs=torch.tensor(obs,dtype=torch.float32).reshape(1,-1).to(device)
                    action=agent.get_action(obs)
                    
                    action=list(action.detach().cpu().numpy().reshape(-1))
                    raw_obs,reward,terminated,truncated,info=env.step(action)
                    # exit(0)
                    done=terminated or truncated
                    if terminated:
                        succ_cnt+=1
                    # env.render()
            print("monster_num",monster_num,"agent_type",agent_type,"succ_rate: ",succ_cnt/max_cnt)
            r_data[f'monster_num_{monster_num}-agent_type_{agent_type}']=  succ_cnt/max_cnt
    
    # json_str = json.dumps(r_data, indent=4, ensure_ascii=False) 
    # with open('./results/succ_rate.json', 'w',encoding='utf-8') as f:
    #     f.write(json_str)

# action0 [ 0.21263161 -1.         -0.22577918  1.          1.7014441  -0.594575
#  -1.          0.1321827   0.          0.5785985  -0.91348875 -1.
#   0.        ]

# action1= [ 0.21263161 -1.         -0.22577918  1.          1.7014441  -0.594575
#  -1.          0.1321827   0.          0.5785985  -0.91348875 -1.
#   0.          0.          0.          0.          0.        ]

