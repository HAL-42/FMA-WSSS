#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/30 20:39
@File    : save_best_·95mask.py
@Software: PyCharm
@Desc    : 
"""
from functools import partial

from alchemy_cat.contrib.voc import label_map2color_map
from alchemy_cat.py_tools import Config, IL

from libs.seeding.mask import thresh_mask

cfg = config = Config()

cfg.eval.seed.save = 'best'

cfg.eval.seed.mask.ini.thresh = 0.95
cfg.eval.seed.mask.cal = IL(lambda c: partial(thresh_mask, **c.eval.seed.mask.ini))
cfg.eval.seed.mask.viz = label_map2color_map
