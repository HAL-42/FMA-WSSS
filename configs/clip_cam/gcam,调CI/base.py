#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/10 12:16
@File    : cfg.py
@Software: PyCharm
@Desc    : 参考configs/clip_cam/调GCAM损/l1_only,amp/cfg.py。
"""
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/clip_cam/base.py', 'configs/clip_cam/_patches/ref/coop_ctx-M=16-V1.py')

# * 关闭分类损失。
cfg.loss.loss_items.multi_cls.weights = 0.
