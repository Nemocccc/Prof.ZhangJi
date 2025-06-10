import torch
from test_agent_1 import Agent as Agent1
from test_agent_0 import Agent as Agent0

model_path0="runs\custom_envs\TVM-v1__train_agent__1__1739801848/train_agent_final.cleanrl_model"
model_path1="modified_model.pth"

agent0=Agent0()
agent1=Agent1()


agent0.load_state_dict(torch.load(model_path0,map_location="cpu"))
agent1.load_state_dict(torch.load(model_path1,map_location="cpu"))

x = torch.tensor([
    0.8238641, 8.83154952, -0.12982783, 0.36119124, 3.303411, 5.00524454,
    -0.15135623, 0.72151056, 8.14356303, 0.0612181, -0.76452168, 0.61166984,
    0.0, 1.0, 0.0, 0.0, 5.0, 0.0, 0.0
])

print(agent0.get_action(x))

print(agent1.get_action(x))