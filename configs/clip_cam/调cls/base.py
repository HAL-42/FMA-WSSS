#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/28 11:07
@File    : base.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/clip_cam/base.py', 'configs/clip_cam/_patches/ref/coop_ctx-M=16-V1.py')

cfg.loss.loss_items.cam_lb.weights = (0., 0.)  # CAM上损失置0。
cfg.auger.train.ini.ol_cls_lb = False
