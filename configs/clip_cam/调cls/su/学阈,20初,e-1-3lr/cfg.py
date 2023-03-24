#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/24 0:08
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Cfg2Tune, Param2Tune

cfg = config = Cfg2Tune('configs/patterns/seg_metrics/cls_m_IoU,pra.py',
                        cfgs_update_at_parser=('configs/clip_cam/调cls/su/base.py',))

cfg.rslt_dir = ...

cfg.rand_seed = 0

cfg.loss.loss_items.multi_cls.ini.thresh_lr = Param2Tune([0.001, 0.1])
