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

# * 一律预热500迭代（0.75）轮，训练5,10,15,20轮。
cfg.solver.max_iter = Param2Tune([500 + 3300, 500 + 6600, 500 + 9900, 500 + 13200, 500 + 16500, 500 + 19800])
