#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/29 14:18
@File    : scale_crf.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config, IL

cfg = config = Config('configs/patterns/crf/base.py')

cfg.crf.scale = 1

cfg.crf.ini.iter_max = 10
cfg.crf.ini.pos_w = 3
cfg.crf.ini.pos_xy_std = IL(lambda c: 3 / c.crf.scale, priority=-1)
cfg.crf.ini.bi_w = 10
cfg.crf.ini.bi_xy_std = IL(lambda c: 80 / c.crf.scale, priority=-1)
cfg.crf.ini.bi_rgb_std = 13
