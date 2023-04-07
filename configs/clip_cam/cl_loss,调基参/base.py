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

from libs.loss.multi_cls.cl_loss import MultiLabelCLLoss

cfg = config = Config('configs/clip_cam/base.py',
                      'configs/clip_cam/_patches/ref/coop_ctx-M=16-V1.py',
                      'configs/clip_cam/_patches/cls_only.py')

cfg.loss.loss_items.multi_cls.ini.gamma = None
cfg.loss.loss_items.multi_cls.ini.reduce = 'pos_mean'
cfg.loss.loss_items.multi_cls.cri = IL(lambda c: MultiLabelCLLoss(**c.loss.loss_items.multi_cls.ini))
cfg.loss.loss_items.multi_cls.cal = lambda cri, inp, out: cri(out.fg_logits, inp.fg_cls_lb)
cfg.loss.loss_items.multi_cls.weights = 1
