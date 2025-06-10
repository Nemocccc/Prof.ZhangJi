## bydrill_v2.1.5.2
### 环境依赖

python版本为3.9.12，用户可以根据操作系统选择对应版本的drill。

下载指定版本drill后，通过 pip install -r requirements.txt 下载drill依赖，并将drill包添加到当前python环境中。

### 本地测试

拉取bydrill测试样例
(工程地址 : https://gitlab.inspir.work/Tutorial/f4v1.git 分支 : v2.1.5.2)。

运行local_test测试工程是否正确运行，本地运行共有两种模式。其中，inference模式代表只运行智能体前向推理，debug模式则对神经网络训练逻辑进行测试。

如inference和debug测试均通过，则可使用训练云对智能体进行训练。

### 智能体训练与模型加载

训练云相关操作详见训练云线上使用文档。

在训练好的任务中下载指定模型（npz），并通过修改local_test中BPLocalRunner的model_info_dict参数加载训练好的模型。