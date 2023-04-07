#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/4/6 21:17
@File    : cls_only.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config()

cfg.loss.loss_items.cam_lb.weights = (0., 0.)  # CAM上损失置0。
cfg.auger.train.ini.ol_cls_lb = False
