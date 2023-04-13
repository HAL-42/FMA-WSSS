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
from alchemy_cat.py_tools import Config, IL

cfg = config = Config()

voc_ratio = 500 / 361  # VOC数据集55%图片为500x375（总是长边在前），28%为500x334，加权平均，得到500x361。

cfg.auger.train.ini.scale_crop_method.set_whole(True)
cfg.auger.train.ini.scale_crop_method.method = 'rand_range'
cfg.auger.train.ini.scale_crop_method.low_size = IL(lambda c:
                                                    round(c.auger.train.ini.scale_crop_method.short_thresh * voc_ratio))
cfg.auger.train.ini.scale_crop_method.high_size = IL(lambda c:
                                                     round((c.auger.train.ini.scale_crop_method.crop_size ** 2 /
                                                            c.auger.train.min_area_propor * voc_ratio) ** 0.5))
cfg.auger.train.ini.scale_crop_method.short_thresh = IL(lambda c: c.auger.train.ini.scale_crop_method.crop_size,
                                                        priority=0)
cfg.auger.train.ini.scale_crop_method.crop_size = ...
cfg.auger.train.min_area_propor = ...
