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

cfg = config = Config('configs/clip_cam/调cls/cl_loss/infer/5100/aff2次,at_score,调crf/base.py',
                      'configs/patterns/crf/clip_es_crf,no_pp.py')

cfg.rslt_dir = ...

# * aff两次时，CRF一般在p=1时最优。
cfg.eval.seed.bg_methods = [{'method': 'pow', 'pow': p} for p in range(1, 2)]
