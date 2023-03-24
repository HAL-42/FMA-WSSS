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
from alchemy_cat.py_tools import Cfg2Tune, Param2Tune

cfg = config = Cfg2Tune('configs/patterns/seg_metrics/cls_m_IoU,pra.py',
                        cfgs_update_at_parser=('configs/clip_cam/base.py',
                                               'configs/clip_cam/_patches/few_shot.py',
                                               'configs/clip_cam/_patches/l2_only.py'))

cfg.rslt_dir = ...

cfg.rand_seed = Param2Tune([10, 20, 30, 40])

# * 设置样本数为32。
cfg.dt.few_shot.ini.shot_num = 32

# * 一律预热1.5轮，训练50轮。
cfg.sched.warm.warm_epoch = 1.5
cfg.solver.max_epoch = 50
