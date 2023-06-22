#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/2/21 20:41
@File    : par_run.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import os
import os.path as osp
import pickle
import subprocess
import sys
from typing import Dict, Any

from alchemy_cat.py_tools.param_tuner import Cfg2TuneRunner
from alchemy_cat.torch_tools import allocate_cuda_by_group_rank

sys.path = ['.', './src'] + sys.path  # noqa: E402


class TrainCoOp(Cfg2TuneRunner):

    @staticmethod
    def work(pkl_idx_cfg_pkl_cfg_rslt_dir):

        pkl_idx, (cfg_pkl, cfg_rslt_dir) = pkl_idx_cfg_pkl_cfg_rslt_dir

        # * 根据分到的配置，训练网络。
        if (not args.purge) and osp.isfile(final_pth := osp.join(cfg_rslt_dir, 'checkpoints', 'final.pth')):
            print(f"{final_pth}存在，跳过{cfg_pkl}。")
        else:
            # * 找到当前应当使用的CUDA设备，并等待当前CUDA设备空闲。
            _, env_with_current_cuda = allocate_cuda_by_group_rank(pkl_idx, 1, block=True, verbosity=True)

            # * 在当前设备上执行训练。
            subprocess.run([sys.executable, 'src/tasks/train_coop/run.py',
                            '-i', f'{args.infer_only}',
                            '-e', f'{args.eval_only}',
                            '--no_cache', f'{args.no_cache}',
                            '-c', cfg_pkl],
                           check=False, env=env_with_current_cuda)

    def gather_metric(self, cfg_rslt_dir, run_rslt, param_comb) -> Dict[str, Any]:
        with open(osp.join(cfg_rslt_dir, 'infer', 'final', 'eval', 'statistics.pkl'), 'rb') as pkl_f:
            metrics = pickle.load(pkl_f)
        return {name: metrics[name] for name in self.metric_names}


parser = argparse.ArgumentParser()
parser.add_argument('--purge', default=0, type=int)
parser.add_argument('-c', '--config', type=str)
parser.add_argument("-i", '--infer_only', default=0, type=int)
parser.add_argument("-e", '--eval_only', default=0, type=int)
parser.add_argument('--no_cache', default=0, type=int)
args = parser.parse_args()

runner = TrainCoOp(args.config,
                   config_root='configs',
                   experiment_root='experiment',
                   pool_size=len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
runner.tuning()
