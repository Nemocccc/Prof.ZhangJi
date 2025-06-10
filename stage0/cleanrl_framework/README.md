# 文件说明
python 版本3.9.21

pip配置环境时可以更换清华源

test_agent.py 测试智能体

test_tvm.py 测试场景

train_agent.py 训练智能体ppo代码

videos 训练过程中的视频

runs tensorboard存储文件夹，训练曲线，查看命令 tensorboard --logdir=./runs

custom_envs 自定义环境

cleanrl、cleanrl_utils 强化学习训练框架cleanrl

rl_plotter rl绘图包

results 测试结果存储

# rl_plotter

绘图包

```
python ./rl_plotter/plotter.py 
 --show --filters logs\PPO_TVM_v0 --title PPO_TVM_v0 --filename episode_length.csv --ylabel "Episode Length"
```

```
python ./rl_plotter/plotter.py 
 --show --filters logs\PPO_TVM_v0 --title PPO_TVM_v0 --filename episode_reward.csv
```

# 环境迁移

- 联网机器

生成需求文件
```
pip freeze > requirements.txt
```

下载wheel
```
python3 -m pip download -r requirements.txt -d packages --no-deps
```

- 非联网机器

1. 在~/anaconda3/envs/目录下，新建一个空文件夹，目录名为环境名，使用conda env list 查看环境列表时并不会显示该环境名称；这里我创建一个test环境名。

2. 但是可以进入新建的环境：使用conda activate +环境名进入新环境；

3. 然后在当前环境下安装Python：conda install offline_trans\conda_pkgs\python-3.9.18-h6244533_0.tar.bz2 

有时候会报错：
requests.exceptions.ConnectionError: HTTPSConnectionPool(host=‘repo.anaconda.com’, port=443): Max retries exceeded with url: /pkgs/main/notices.json (Caused by NewConnectionError(‘<urllib3.connection.HTTPSConnection object at 0x7fb9a8912dc0>: Failed to establish a new connection: [Errno 101] Network is unreachable’))
可以使用命令conda install offline_trans\conda_pkgs\python-3.9.18-h6244533_0.tar.bz2 --offline 来解决

4. 在新的环境中可以看到安装的Python，运行python --version

5. 安装pip， conda install offline_trans\conda_pkgs\python-3.9.18-h6244533_0.tar.bz2

6. 此时安装的pip位于新建环境中，输入pip --version 查看pip应位于新建环境目录下。

7. 导入第三方库
```
pip install --no-index --find-links=offline_trans\packages -r requirements.txt   
```


需要测试：test_cuda.py, test_agent.py, test_tvm.py, train_agent.py, 终端tensorboard --logdir=./runs
