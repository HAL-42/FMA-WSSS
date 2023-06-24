#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/3 18:02
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Cfg2Tune, Param2Tune

cfg = config = Cfg2Tune('configs/patterns/seg_metrics/cls_m_IoU,pra.py',
                        cfgs_update_at_parser=('configs/clip_cam/coco,cl_loss/scale/base.py',))

cfg.rslt_dir = ...

# * 设置不同的随机初始种子。
cfg.model.initialize_seed = Param2Tune(['真'])

# * 设置数据增强。
cfg.auger.train.ini.scale_crop_method.crop_size = 224
cfg.auger.train.ini.scale_crop_method.high_low_ratio = 1.67

# * 设置模型初始化。
cfg.model.ini.ctx_cfg.ctx_std = 0.015

# * 设置不同的迭代次数。
cfg.solver.max_iter = Param2Tune([4000, 6000, 8000, 30000])
