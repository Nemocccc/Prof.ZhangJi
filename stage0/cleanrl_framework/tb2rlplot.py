'''
生成两组带有噪音的log_t(x)的曲线
'''
from rl_plotter.logger import Logger
import random,math
import pandas as pd

 
exp_name = 'PPO'
#第一个算法的第一次实验曲线
logger = Logger(exp_name=exp_name, env_name='TVM-v0')
df=pd.read_csv('results/tb_csvs/run-custom_envs_TVM-v0__train_agent__1__1734418278-tag-charts_episodic_length.csv')
for i in range(0,len(df)):
    logger.update(score=[df['Value'].iloc[i]], total_steps=df['Step'].iloc[i])
