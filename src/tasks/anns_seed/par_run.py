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

from alchemy_cat.py_tools.param_tuner import Cfg2TuneRunner
from alchemy_cat.torch_tools import allocate_cuda_by_group_rank

sys.path = ['.', './src'] + sys.path  # noqa: E402


class SegmentSeed(Cfg2TuneRunner):

    @staticmethod
    def work(pkl_idx_cfg_pkl_cfg_rslt_dir):

        pkl_idx, (cfg_pkl, cfg_rslt_dir) = pkl_idx_cfg_pkl_cfg_rslt_dir

        # * 根据分到的配置，运行任务。
        if (not args.purge) and osp.isfile(eval_metric := osp.join(cfg_rslt_dir, 'eval', 'statistics.pkl')):
            print(f"{eval_metric}存在，跳过{cfg_pkl}。")
        else:
            # * 找到当前应当使用的CUDA设备，并等待当前CUDA设备空闲。
            _, env_with_current_cuda = allocate_cuda_by_group_rank(pkl_idx, 1, block=False, verbosity=True)

            # * 在当前设备上执行训练。
            subprocess.run([sys.executable, 'src/tasks/anns_seed/run.py',
                            '-c', cfg_pkl],
                           check=False, env=env_with_current_cuda)

    def gather_metric(self, cfg_rslt_dir, run_rslt, param_comb) -> dict[str, ...]:
        with open(osp.join(cfg_rslt_dir, 'eval', 'statistics.pkl'), 'rb') as pkl_f:
            metrics = pickle.load(pkl_f)
        return {name: metrics[name] for name in self.metric_names}


parser = argparse.ArgumentParser()
parser.add_argument('--purge', default=0, type=int)
parser.add_argument('-c', '--config', type=str)
args = parser.parse_args()

runner = SegmentSeed(args.config,
                     config_root='configs',
                     experiment_root='experiment',
                     pool_size=len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
runner.tuning()
