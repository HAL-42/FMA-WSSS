#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/23 23:20
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config, IL

from libs.loss.multi_cls.bce_loss import ScaledASL

cfg = config = Config('configs/clip_cam/调cls/base.py')

cfg.loss.loss_items.multi_cls.ini.s = 1.
cfg.loss.loss_items.multi_cls.ini.b = -20.
cfg.loss.loss_items.multi_cls.ini.proj_lr = IL(lambda c: c.opt.base_lr, priority=0)
cfg.loss.loss_items.multi_cls.ini.norm_s = True
cfg.loss.loss_items.multi_cls.ini.ASL_cfg.gamma_neg = 0  # 等价标准MultiLabelSoftMarginLoss
cfg.loss.loss_items.multi_cls.ini.ASL_cfg.gamma_pos = 0
cfg.loss.loss_items.multi_cls.ini.ASL_cfg.clip = 0
cfg.loss.loss_items.multi_cls.ini.ASL_cfg.disable_torch_grad_focal_loss = True
cfg.loss.loss_items.multi_cls.cri = IL(lambda c: ScaledASL(**c.loss.loss_items.multi_cls.ini))
cfg.loss.loss_items.multi_cls.cal = lambda cri, inp, out: cri(out.fg_logits, inp.fg_cls_lb)
cfg.loss.loss_items.multi_cls.weights = 1
