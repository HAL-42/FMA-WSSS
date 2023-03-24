#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/10 12:16
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Cfg2Tune, Param2Tune, IL

cfg = config = Cfg2Tune('configs/patterns/seg_metrics/cls_m_IoU,pra.py',
                        cfgs_update_at_parser=('configs/clip_cam/gcam,l1_only,小样/base.py',))

cfg.rslt_dir = ...

cfg.rand_seed = 0

# * 设置不同的样本数。
# 分别为全数据集的3.2%, 6.4%, 12.7%, 25.4%, 50.8%, 76.3%。
cfg.dt.few_shot.ini.shot_num = Param2Tune([16, 32, 64, 128, 256, 384])

# * 一律预热1.5轮，训练50轮。
def warm_iters(c):  # noqa
    return round(c.dt.few_shot.epoch_len * 1.5)
cfg.sched.warm.warm_iters = IL(warm_iters, priority=0)  # noqa 1.5轮预热
def max_iter(c):  # noqa
    return round(c.dt.few_shot.epoch_len * 51.5)
cfg.solver.max_iter = IL(max_iter, priority=0)  # noqa 1.5轮预热 + 50轮训练
