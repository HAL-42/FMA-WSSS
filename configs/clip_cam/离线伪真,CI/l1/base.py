#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/4/13 23:19
@File    : base.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/clip_cam/base.py', 'configs/clip_cam/_patches/ref/coop_ctx-M=16-V1.py')

# * 关闭分类损失。
cfg.loss.loss_items.multi_cls.weights = 0.
