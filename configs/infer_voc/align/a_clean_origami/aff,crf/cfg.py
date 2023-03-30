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
                      'configs/patterns/crf/deeplab_crf.py',
                      'configs/patterns/crf/crf_eval.py')

cfg.rslt_dir = ...

# * 和aff结果相同，不用在可视化。
cfg.solver.viz_cam = False
cfg.solver.viz_score = False
