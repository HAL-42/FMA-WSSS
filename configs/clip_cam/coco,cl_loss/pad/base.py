#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/4/6 21:19
@File    : base.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config, IL

cfg = config = Config('configs/clip_cam/coco,cl_loss/base.py',  # COCO+多标签分类损失。
                      'configs/patterns/aug/coco_scale_long_pad.py')  # 缩放+pad增强。

# * 设置数据加载。
cfg.loader.train.sub_iter_num = 4

# * 设置缩放增强。
cfg.auger.train.ini.scale_crop_method.low_size = 384  # 320 × 1.2
cfg.auger.train.ini.scale_crop_method.high_size = IL(lambda c: c.auger.train.ini.scale_crop_method.low_size)
