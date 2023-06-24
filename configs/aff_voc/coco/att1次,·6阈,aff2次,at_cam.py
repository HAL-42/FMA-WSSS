#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/6/24 18:19
@File    : att1次,·6阈,aff2次,at_cam.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/aff_voc/coco/base.py')

# * 调整参数。
cfg.aff.ini.att2aff_cfg.method.n_iter = 1
cfg.aff.ini.aff_mask_cfg.method.thresh = .6
cfg.aff.ini.aff_cfg.n_iters = 2
cfg.aff.ini.aff_at = 'cam'

# * 设定eval方法。
cfg.eval.seed.bg_methods = [{'method': 'pow', 'pow': p} for p in range(1, 3)]
