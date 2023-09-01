#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/29 16:14
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config
from libs.data.coco_dt import COCO

cfg = config = Config('configs/aff_voc/coco/base.py',
                      'configs/patterns/seed/save_best_·95mask.py')

cfg.solver.viz_cam = False

# * 调整参数。
cfg.aff.ini.att2aff_cfg.method.n_iter = 1  # noqa
cfg.aff.ini.aff_mask_cfg.method.thresh = .6
cfg.aff.ini.aff_cfg.n_iters = 1
cfg.aff.ini.aff_at = 'cam'

# * 设定eval方法。
cfg.eval.seed.bg_methods = [{'method': 'pow', 'pow': p} for p in range(3, 4)]

cfg.eval.seed.mask.viz = COCO.label_map2color_map
