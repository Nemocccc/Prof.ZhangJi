import argparse
from drill.local import BPLocalRunner
from configs.training_config import builder, algo

parser = argparse.ArgumentParser()
parser.add_argument("--mode", "-m", help="'debug' or 'inference'", default="inference", type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    # 创建本地前向推理实例
    runner = BPLocalRunner(builder, algorithm=algo)
    # runner = BPLocalRunner(builder, model_info_dict={"f4v1_model":
    #                                                      {"model_file": "./models/f4v1_model/f4v1_model_0.npz",
    #                                                       "backend": 'tensorflow'}, })

    # debug模式用于测试态势数据格式以及神经网络结构；inference模式用于本地前向推理
    if args.mode == "debug":
        stats = runner.debug(return_stats=True)
        print(stats)
    elif args.mode == "inference":
        # 通过设置一个很大的step_num来实现前向多局episode
        runner.evaluate(max_step_num=10000)
        # runner.evaluate(max_step_num=10000, is_batch_inference=False)
