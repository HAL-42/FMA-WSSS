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

# * 设置l1和l2损失。
cfg.loss.loss_items.cam_lb.ini.loss_type = Param2Tune(['l1', 'l2'])

# * 设置不同的前背景平均方法。
# 分别为前背景平衡、前背景且主样本平衡。
cfg.loss.loss_items.cam_lb.ini.reduce = Param2Tune(['all', 'fg-bg', 'fg-bg-per-pos'])

# * 设置max是否detach。
cfg.loss.loss_items.cam_lb.ini.detach_max = Param2Tune([True, False])
