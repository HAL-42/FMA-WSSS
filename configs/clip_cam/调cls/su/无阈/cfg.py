#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/24 0:08
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/clip_cam/调cls/su/base.py')

cfg.loss.loss_items.multi_cls.ini.thresh = None
cfg.loss.loss_items.multi_cls.ini.thresh_lr = None
