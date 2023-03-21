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
                        cfgs_update_at_parser=('configs/clip_cam/调GCAM损/l2_only,amp,调fb权/base.py',))

cfg.rslt_dir = ...

# * 设置不同的前背景平均方法。
# 分别为前背景平衡、前背景且主样本平衡。
cfg.loss.loss_items.cam_lb.ini.reduce = Param2Tune(['fg-bg', 'fg-bg-per-pos'])
