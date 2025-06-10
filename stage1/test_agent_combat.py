# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
# 启用同步 CUDA 错误报告和设备端断言
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import custom_envs

import torch.nn.functional as F

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Combat-v0"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        # 确保嵌入维度可以被头数整除
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"
        
        # 线性变换层
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        # 输出线性层
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        query, key, value=x,x,x
        # 输入形状: (batch_size, seq_len, embed_dim)
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # 线性变换
        Q = self.q_linear(query)  # (batch_size, seq_len, embed_dim)
        K = self.k_linear(key)    # (batch_size, seq_len, embed_dim)
        V = self.v_linear(value)  # (batch_size, seq_len, embed_dim)
        
        # 分割成多个头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力得分 (点积注意力)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, num_heads, seq_len, seq_len)
        attn_scores = attn_scores / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # 应用掩码（可选）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)  # 应用dropout
        
        # 计算加权的值
        output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)
        
        # 拼接多个头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)  # (batch_size, seq_len, embed_dim)
        
        # 输出线性层
        output = self.out_linear(output)
        
        return output



def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.entity_encoder = nn.Sequential(
            layer_init(nn.Linear(3, 64)),
            nn.ReLU(),            
            MultiHeadAttention(64,4),
            nn.ReLU(),
            layer_init(nn.Linear(64, 32)),
            nn.ReLU(),
        )
        self.common_encoder= nn.Sequential(
            layer_init(nn.Linear(3*3*2+4, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 32)),
            nn.ReLU(),
        )
        
        self.aggregator=nn.Sequential(
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
        )

        self.critic = layer_init(nn.Linear(64, 1), std=1.0)

        self.actor = layer_init(nn.Linear(64, 5), std=0.01)


    def get_value(self, x):

        batch_size=x.shape[0]
        entity_obs=x[:,22:].reshape(batch_size,-1,3)
        common_obs=x[:,:22]
        entity_fea=self.entity_encoder(entity_obs)[:,-1,:].squeeze(1)
        common_fea=self.common_encoder(common_obs)
        cat_fea=torch.cat([entity_fea,common_fea],dim=1)
        x=self.aggregator(cat_fea)

        return self.critic(x)

    def get_action_and_value(self, x, action=None):

        batch_size=x.shape[0]
        entity_obs=x[:,22:].reshape(batch_size,-1,3)
        common_obs=x[:,:22]
        entity_fea=self.entity_encoder(entity_obs)
        entity_fea, indices = torch.max(entity_fea, dim=1)
        common_fea=self.common_encoder(common_obs)
        cat_fea=torch.cat([entity_fea,common_fea],dim=1)
        x=self.aggregator(cat_fea)
        # print(x.shape)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_action(self, x):
        with torch.no_grad():
                
            batch_size=x.shape[0]
            entity_obs=x[:,22:].reshape(batch_size,-1,3)
            common_obs=x[:,:22]
            entity_fea=self.entity_encoder(entity_obs)[:,-1,:].squeeze(1)
            # entity_fea, indices = torch.max(entity_fea, dim=1)
            common_fea=self.common_encoder(common_obs)
            cat_fea=torch.cat([entity_fea,common_fea],dim=1)
            x=self.aggregator(cat_fea)
            # print(x.shape)
            logits = self.actor(x)
            probs = Categorical(logits=logits)
            # print(logits)
            action = torch.argmax(logits, dim=1)  # 默认是 dim=-1
        return action


if __name__ == "__main__":
    args = tyro.cli(Args)
    n_agents=4

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.make(args.env_id, n_agents=n_agents,render_mode="human")

    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load("runs\Combat-v0__train_agent_combat__1__1740324754/train_agent_combat_150.cleanrl_model"))
    
    done=[False]*n_agents
    obs,_=envs.reset()
    obs = torch.Tensor(obs).to(device)
    while sum(done)!=n_agents:
        actions=list(agent.get_action(obs).cpu().numpy().flatten())
        print(actions)
        obs,reward,te,tr,info=envs.step(actions)
        obs = torch.Tensor(obs).to(device)
        done=te

        envs.render()
