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
                      'configs/patterns/crf/crf_eval.py')

cfg.rslt_dir = ...

# * 和aff结果相同，不用在可视化。
cfg.solver.viz_cam = False
cfg.solver.viz_score = False

# * aff score。
cfg.aff.ini.aff_at = 'score'

# * 设置aff次数。
cfg.aff.ini.aff_cfg.n_iters = 2
cfg.aff.ori_cam_dir = 'experiment/clip_cam/调cls/cl_loss/infer/5100/cam'

# * aff两次时，CRF一般在p=1时最优。
cfg.eval.seed.bg_methods = [{'method': 'pow', 'pow': p} for p in range(1, 3)]
