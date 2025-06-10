import json
import custom_envs
import gymnasium as gym
from train_agent import *
from PIL import Image
import os
import shutil

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
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space.shape)))

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


if __name__ == "__main__":

    # TRY NOT TO MODIFY: seeding
    args = tyro.cli(Args)
    # args.capture_video=False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")



    record=2 # 是否进行对局信息记录

    if record==0:
        # env setup
        env =gym.make("custom_envs/TVM_re-v0", render_mode="rgb_array")
        env = gym.wrappers.ClipAction(env)
            
        agent = Agent(env).to(device)
        model_path="./runs/custom_envs/TVM-v0__train_agent__1__1734418278/train_agent_600.cleanrl_model"
        agent.load_state_dict(torch.load(model_path))
    
        clear_directory("./results/imgs")
        clear_directory("./results/js")

        done=False
        obs,info=env.reset(seed=26)


        r_data={}
        r_data[f'{env.decision_step}']=info.copy()
        while True:
            image=env.render()
            # 使用Pillow库将NumPy数组转换为图像
            image = Image.fromarray(image)  # "L" 表示灰度模式
            # 保存图像
            image.save(f"./results/imgs/img_{env.decision_step}.png")
            
            obs=torch.tensor(obs,dtype=torch.float32).reshape(1,-1).to(device)
            action=agent.actor_mean(obs)
            action=list(action.detach().cpu().numpy().reshape(-1))
            obs,reward,terminated,truncated,info=env.step(action)
            done=terminated or truncated
            

            r_data[f'{env.decision_step}']=info.copy()
            if done:
                break
            

        json_str = json.dumps(r_data, indent=4, ensure_ascii=False) 
        with open('./results/js/result_record.json', 'w',encoding='utf-8') as f:
            f.write(json_str)
    elif record==1:
        # env setup
        env =gym.make("custom_envs/TVM_re-v0", render_mode="rgb_array")
        env = gym.wrappers.ClipAction(env)
            
        agent = Agent(env).to(device)
        model_path="./runs/custom_envs/TVM-v0__train_agent__1__1734418278/train_agent_600.cleanrl_model"
        agent.load_state_dict(torch.load(model_path))

        r_data={}
        
        for i in range(10):
            succ_cnt=0
            for _ in range(100):
                done=False
                obs,info=env.reset()

                while not done:
                    obs=torch.tensor(obs,dtype=torch.float32).reshape(1,-1).to(device)
                    action=agent.actor_mean(obs)
                    action=list(action.detach().cpu().numpy().reshape(-1))
                    obs,reward,terminated,truncated,info=env.step(action)
                    done=terminated or truncated
                    if terminated:
                        succ_cnt+=1
                    # env.render()
            print("succ_rate: ",succ_cnt/100)
            r_data[f'test_{i}']=  succ_cnt/100
        
        json_str = json.dumps(r_data, indent=4, ensure_ascii=False) 
        with open('./results/js/succ_rate.json', 'w',encoding='utf-8') as f:
            f.write(json_str)
    else:
        # env setup
        env =gym.make("custom_envs/TVM_re-v0")
        env = gym.wrappers.ClipAction(env)
            
        agent = Agent(env).to(device)
        model_path="./runs/custom_envs/TVM-v0__train_agent__1__1734418278/train_agent_600.cleanrl_model"
        agent.load_state_dict(torch.load(model_path))
        done=False
        obs,info=env.reset()

        while not done:
            obs=torch.tensor(obs,dtype=torch.float32).reshape(1,-1).to(device)
            action=agent.actor_mean(obs)
            action=list(action.detach().cpu().numpy().reshape(-1))
            obs,reward,terminated,truncated,info=env.step(action)
            done=terminated or truncated
            env.render()