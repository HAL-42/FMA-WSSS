#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/10 21:27
@File    : cfg.py
@Software: PyCharm
@Desc    :
"""
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/infer_voc/base_coco.py')

# * 输入一律缩放到320x320。
cfg.auger.val.ini.scale_crop_method = 320

# * 因为输入尺寸相同，batch可以不为1。
cfg.loader.val.batch_size = 4
