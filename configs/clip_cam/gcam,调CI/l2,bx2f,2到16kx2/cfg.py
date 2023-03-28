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
                        cfgs_update_at_parser=('configs/clip_cam/gcam,调CI/base.py',))

cfg.rslt_dir = ...

# * 设置l2损失。
cfg.loss.loss_items.cam_lb.ini.loss_type = 'l2'

# * 设置l2损失的权重。
cfg.loss.loss_items.cam_lb.weights = Param2Tune([(2000, 4000), (4000, 8000), (8000, 16000), (16000, 32000)])

# * 对l2损失，设置val时的bg power为为2和4。
cfg.val.cfg.eval.seed.bg_methods = [{'method': 'pow', 'pow': p} for p in (2, 4)]
