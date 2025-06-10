import os
import sys

path = os.path.abspath(os.path.dirname(__file__))
if 'example' in path:
    while not path.endswith('example'):
        path = os.path.dirname(path)
path = os.path.dirname(path)
sys.path.insert(0, path)

import argparse

from example.dueling.cartpole.configs.training_config import builder, algo
from lib.local import BPLocalRunner

parser = argparse.ArgumentParser()
parser.add_argument("--mode", "-m", help="'debug' or 'inference'", default="debug", type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    runner = BPLocalRunner(builder, algorithm=algo)
    # runner = BPLocalRunner(builder, algorithm=algo,
    #                        model_info_dict={"atari_model": {"model_file": "./5000.npz",
    #                                                         "backend": 'tensorflow'}})

    if args.mode == "debug":
        stats = runner.debug(return_stats=True)
        print(stats)
    elif args.mode == "inference":
        # 通过设置一个很大的step_num来实现前向多局episode
        runner.evaluate(max_step_num=10000)
        # runner.evaluate(max_step_num=10000, is_batch_inference=False)
