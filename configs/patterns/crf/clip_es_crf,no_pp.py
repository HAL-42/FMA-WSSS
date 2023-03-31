#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/29 14:18
@File    : clip_es_crf.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/patterns/crf/base.py')

cfg.crf.ini.iter_max = 10
cfg.crf.ini.pos_w = 3
cfg.crf.ini.pos_xy_std = 1
cfg.crf.ini.bi_w = 4
cfg.crf.ini.bi_xy_std = 67
cfg.crf.ini.bi_rgb_std = 3
