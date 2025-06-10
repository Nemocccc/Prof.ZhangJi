import torch

def change_model_weights_and_biases(model_path, output_path):
    # 加载.pth文件
    model_state_dict = torch.load(model_path, map_location="cpu")

    # 修改指定层的权重和偏置
    model_state_dict['actor_mean.4.weight'] = torch.cat([model_state_dict['actor_mean.4.weight'], torch.randn(4, 64)], dim=0)
    model_state_dict['actor_mean.4.bias'] = torch.cat([model_state_dict['actor_mean.4.bias'], torch.randn(4)], dim=0)

    # 修改 actor_logstd 层
    model_state_dict['actor_logstd'] = torch.cat([model_state_dict['actor_logstd'], torch.randn(1, 4)], dim=1)

    # 打印修改后的各层权重和偏置形状
    for key, value in model_state_dict.items():
        print(f"Layer: {key}")
        print(f"Shape: {value.shape}")
        print("-" * 30)

    # 保存修改后的模型权重到新的.pth文件
    torch.save(model_state_dict, output_path)
    print(f"Modified model saved to {output_path}")

# 输入和输出路径
model_path = "runs\TVM-v2__train_agent_tvm__1__1740390195/train_agent_tvm_final.cleanrl_model"
output_path = "./modified_model.pth"  # 替换为你希望保存的修改后的.pth文件路径

# 调用函数
change_model_weights_and_biases(model_path, output_path)
    

