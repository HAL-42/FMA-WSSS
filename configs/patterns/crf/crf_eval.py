#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/29 15:54
@File    : crf_eval.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config, IL

cfg = config = Config()

cfg.eval.seed.crf = IL(lambda c: c.crf.cal)

cfg.eval.seed.bg_methods = [{'method': 'pow', 'pow': p} for p in range(1, 3)]  # CRF后，bg power一般在1时最优。
