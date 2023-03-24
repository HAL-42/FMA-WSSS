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
                        cfgs_update_at_parser=('configs/clip_cam/调cls/cl,·2sm_logits_sharp/base.py',))

cfg.rslt_dir = ...

cfg.rand_seed = 0

cfg.loss.loss_items.sharp.ini.reduce = Param2Tune(['mean', 'batch_mean'])
