#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/4/12 17:00
@File    : voc_rand_range.py
@Software: PyCharm
@Desc    : 
"""
from functools import partial

from alchemy_cat.py_tools import Config, IL
from alchemy_cat.alg import divisible_by_n

cfg = config = Config()

cfg.auger.train.ini.scale_crop_method.set_whole(True)
cfg.auger.train.ini.scale_crop_method.method = 'scale_long_pad'
cfg.auger.train.ini.scale_crop_method.low_size = ...
cfg.auger.train.ini.scale_crop_method.high_size = ...
cfg.auger.train.ini.scale_crop_method.short_thresh = 0
cfg.auger.train.ini.scale_crop_method.aligner = IL(lambda c: partial(divisible_by_n, n=c.model.patch_size))
