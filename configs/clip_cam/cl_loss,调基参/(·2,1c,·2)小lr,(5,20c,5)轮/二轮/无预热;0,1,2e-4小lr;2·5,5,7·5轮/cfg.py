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
                        cfgs_update_at_parser=('configs/clip_cam/cl_loss,调基参/base.py',))

cfg.rslt_dir = ...

# * 最小学习率是初始学习率的0, 1e-4, 2e-4。
cfg.sched.main.ini.eta_min = Param2Tune([0, 0.001 * 0.1, 0.001 * 0.2])

# * 一律预热500迭代（0.75）轮，训练2.5, 5, 7.5轮。
cfg.solver.max_iter = Param2Tune([10 + 1600, 10 + 3300, 10 + 5000])

# * 只预热10个迭代，约等于不预热。
cfg.sched.warm.warm_iters = 10
