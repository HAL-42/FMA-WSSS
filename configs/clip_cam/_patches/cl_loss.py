#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/6/22 20:12
@File    : cl_loss.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config, IL

from libs.loss.multi_cls.cl_loss import MultiLabelCLLoss

cfg = config = Config()

cfg.loss.loss_items.multi_cls.set_whole(True)
cfg.loss.loss_items.multi_cls.ini.gamma = None
cfg.loss.loss_items.multi_cls.ini.reduce = 'pos_mean'
cfg.loss.loss_items.multi_cls.cri = IL(lambda c: MultiLabelCLLoss(**c.loss.loss_items.multi_cls.ini))
cfg.loss.loss_items.multi_cls.cal = lambda cri, inp, out: cri(out.fg_logits, inp.fg_cls_lb)
cfg.loss.loss_items.multi_cls.weights = 1
