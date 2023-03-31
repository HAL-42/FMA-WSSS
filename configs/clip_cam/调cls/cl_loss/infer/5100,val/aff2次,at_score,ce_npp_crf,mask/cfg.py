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

cfg = config = Config('configs/aff_voc/base.py',
                      'configs/patterns/crf/clip_es_crf,no_pp.py',
                      'configs/patterns/crf/crf_eval.py',
                      'configs/patterns/seed/save_best_·95mask.py')

cfg.rslt_dir = ...

cfg.dt.val.ini.split = 'val'

# * 调参，不做可视化。
cfg.solver.viz_cam = False
cfg.solver.viz_score = False

cfg.aff.ini.aff_cfg.n_iters = 2
cfg.aff.ini.aff_at = 'score'

cfg.eval.seed.bg_methods = [{'method': 'pow', 'pow': p} for p in range(1, 2)]

cfg.eval.seed.mask.ini.thresh = 0
