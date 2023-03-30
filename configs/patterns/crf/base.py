#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/29 14:11
@File    : base.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config, IL
from alchemy_cat.data.plugins.augers import identical

from utils.crf import ParDenseCRF

cfg = config = Config()

cfg.crf.ini.iter_max = ...
cfg.crf.ini.pos_w = ...
cfg.crf.ini.pos_xy_std = ...
cfg.crf.ini.bi_w = ...
cfg.crf.ini.bi_xy_std = ...
cfg.crf.ini.bi_rgb_std = ...
cfg.crf.ini.img_preprocess = identical  # RGB, uint8, (H, W, 3)图片。
cfg.crf.ini.align_corner = False
cfg.crf.ini.pool_size = 0

cfg.crf.cal = IL(lambda c:
                 ParDenseCRF(**c.crf.ini),
                 priority=0)
