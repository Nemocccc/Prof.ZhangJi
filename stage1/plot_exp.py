import os
import subprocess

def plot_folder_data(folder_path):
    """
    从指定文件夹中读取所有 CSV 文件，并调用 rl_plotter 绘制曲线。

    参数:
        folder_path (str): 包含 CSV 文件的文件夹路径。
    """
    # 获取文件夹中的所有 CSV 文件
    csv_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        print(f"文件夹 {folder_path} 中未找到 CSV 文件。")
        return

    # 构建 rl_plotter 命令
    command = [
        "python", "rl_plotter/plotter.py",
        "--log_dir", f"{folder_path}",  # 动态传入所有 CSV 文件路径
        "--show",  # 显示图像
        "--avg_group",  # 计算均值曲线
        "--shaded_std",  # 绘制标准差范围
        # "--smooth","50"
    ]

    # 打印命令以便调试
    print("运行命令:", " ".join(command))

    # 调用 rl_plotter
    subprocess.run(command)

# # 设置输入文件夹
# input_folder = "plot_datas/train_enet/"  # 替换为你的文件夹路径

# # 调用函数绘制曲线
# plot_folder_data(input_folder)

# 设置输入文件夹
input_folder = "plot_datas/comparsion/"  # 替换为你的文件夹路径

# 调用函数绘制曲线
plot_folder_data(input_folder)