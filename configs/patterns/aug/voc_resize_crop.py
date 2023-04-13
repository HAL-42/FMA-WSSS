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
from alchemy_cat.py_tools import Config

cfg = config = Config()

voc_ratio = 500 / 361  # VOC数据集55%图片为500x375（总是长边在前），28%为500x334，加权平均，得到500x361。

cfg.auger.train.ini.scale_crop_method.set_whole(True)
cfg.auger.train.ini.scale_crop_method.method = 'rand_resize_crop'
cfg.auger.train.ini.scale_crop_method.size = ...
cfg.auger.train.ini.scale_crop_method.scale = ...
# 1.1是为了：1）若退回center crop，总是直接缩放，不做裁剪。2）有更大概率采样到voc_ratio附近的ratio（该ratio更容易满足大scale）。
cfg.auger.train.ini.scale_crop_method.ratio = ((1 / voc_ratio) / 1.1, voc_ratio * 1.1)
