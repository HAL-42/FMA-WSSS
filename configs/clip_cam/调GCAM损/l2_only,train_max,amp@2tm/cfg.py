#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/10 12:16
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
import os.path as osp

from alchemy_cat.py_tools import Config, IL

cfg = config = Config('configs/clip_cam/base.py')

cfg.rslt_dir = ...
cfg.rand_seed = 0  # 与随机参考使用相同的随机种子。如此相比基线多出随机部分，参考不同基线时，有不同的随机性。

# * 设定随机参考。
cfg.rand_ref.ref_dir = 'experiment/clip_cam/调GCAM损/l2_only,amp'
cfg.rand_ref.rand_copy = IL(lambda c:
                            {'initial context': (osp.join(c.rand_ref.ref_dir, 'checkpoints/start.pth'),
                                                 osp.join(c.rslt_dir, 'checkpoints/start.pth'))})

# * 仅使用L2损失。
cfg.loss.loss_items.cam_lb.ini.loss_type = 'l2'
cfg.loss.loss_items.multi_cls.weights = 0.

# * max参与训练。
cfg.loss.loss_items.cam_lb.ini.detach_max = False
