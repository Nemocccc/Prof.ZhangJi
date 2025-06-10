'''
生成两组带有噪音的log_t(x)的曲线
'''
from rl_plotter.logger import Logger
import random,math
import pandas as pd
import os

def get_csv_files(folder_path):
    """
    获取文件夹下所有 CSV 文件的路径。

    参数:
        folder_path (str): 文件夹路径。

    返回:
        list: 包含所有 CSV 文件路径的列表。
    """
    csv_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files
 
def log_csv_data_to_logger(csv_files, exp_name='Basic Evaluator', env_name='TVM-v0'):
    """
    将多个 CSV 文件中的数据记录到 Logger 中。

    参数:
        csv_files (list): 包含 CSV 文件路径的列表。
        exp_name (str): 实验名称，默认为 'Basic Evaluator'。
        env_name (str): 环境名称，默认为 'TVM-v0'。
    """
    # 初始化 Logger
    

    # 遍历每个 CSV 文件并记录数据
    
    for csv_file in csv_files:
        
        parts = csv_file.split('__')
        # 提取种子部分
        seed_part = parts[2]  # 例如 "42"
        # 提取数字部分
        seed = int(seed_part)
        
        logger = Logger(exp_name=exp_name,env_name=env_name,seed=seed_part)
        df = pd.read_csv(csv_file)  # 读取 CSV 文件
        for i in range(len(df)):
            # logger.update(score=[df['Value'].iloc[i]], total_steps=i+1)
            logger.update(score=[df['Value'].iloc[i]], total_steps=df['Step'].iloc[i])
            

    print("数据已记录到 Logger 中。")

csv_folder_path="./tbcsv/TVM-v0"
exp_name='Rule-based Evaluator'
env_name='TVM-v0'

csv_files=get_csv_files(csv_folder_path)
log_csv_data_to_logger(csv_files,exp_name,env_name)

csv_folder_path="./tbcsv/TVM-v1"
exp_name='Neural Network Evaluator Train'
env_name='TVM-v1'

csv_files=get_csv_files(csv_folder_path)
log_csv_data_to_logger(csv_files,exp_name,env_name)

csv_folder_path="./tbcsv/TVM-v2"
exp_name='Neural Network Evaluator'
env_name='TVM-v2'

csv_files=get_csv_files(csv_folder_path)
log_csv_data_to_logger(csv_files,exp_name,env_name)



 
