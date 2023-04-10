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

# * 最小学习率是初始学习率的0.05, 0.1, 0.2。
cfg.sched.main.ini.eta_min = Param2Tune([0.001 * 0.05, 0.001 * 0.1, 0.001 * 0.2])

# * 一律预热500迭代（0.75）轮，训练2.5, 5轮。
cfg.solver.max_iter = Param2Tune([500 + 1600, 500 + 3300])
