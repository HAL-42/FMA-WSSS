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
