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

coco_ratio = 1.42  # VOC数据集平均边长为[618.56, 443.19]，std在50~60。

cfg.auger.train.ini.scale_crop_method.set_whole(True)
cfg.auger.train.ini.scale_crop_method.method = 'rand_range'
# * 核心控制参数。
cfg.auger.train.ini.scale_crop_method.crop_size = ...
cfg.auger.train.ini.scale_crop_method.high_low_ratio = ...
# * 缩放参数。
# 短边下限为crop_size。
cfg.auger.train.ini.scale_crop_method.short_thresh = IL(lambda c: c.auger.train.ini.scale_crop_method.crop_size,
                                                        priority=0)
# 长边下限为短边下限 * coco_ratio。
cfg.auger.train.ini.scale_crop_method.low_size = IL(lambda c:
                                                    round(c.auger.train.ini.scale_crop_method.short_thresh *
                                                          coco_ratio))
# 长边上限为下限 * high_low_ratio。
cfg.auger.train.ini.scale_crop_method.high_size = IL(lambda c:
                                                     round(c.auger.train.ini.scale_crop_method.short_thresh *
                                                           coco_ratio *
                                                           c.auger.train.ini.scale_crop_method.high_low_ratio)
                                                     )
