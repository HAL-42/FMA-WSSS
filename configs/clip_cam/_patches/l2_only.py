#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/22 12:24
@File    : patch_l2_only.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config()

# * 只开启L2损失。
cfg.loss.loss_items.cam_lb.ini.loss_type = 'l2'
cfg.loss.loss_items.multi_cls.weights = 0.
