#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/6/8 21:00
@File    : ann_prob.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config()

cfg.seed.norm_firsts = ['no_norm']
cfg.seed.bg_methods = [{'method': 'alpha_bg', 'alpha': 1},
                       {'method': 'alpha_bg', 'alpha': 2},
                       {'method': 'alpha_bg', 'alpha': 3},
                       {'method': 'alpha_bg', 'alpha': 4},
                       {'method': 'alpha_bg', 'alpha': 5}]
