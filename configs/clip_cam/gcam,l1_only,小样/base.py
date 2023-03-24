#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/16 21:37
@File    : base.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config, IL

from libs.data import FewShotDt

cfg = config = Config('configs/clip_cam/base.py', 'configs/clip_cam/_patches/ref/coop_ctx-M=16-V1.py')

cfg.rslt_dir = ...

# * 设定小样本集。
dt = cfg.dt

dt.few_shot.ini.shot_num = ...
dt.few_shot.ini.seed = IL(lambda c: c.rand_seed, priority=-10)
dt.few_shot.ini.except_bg = False
dt.few_shot.dt = IL(lambda c:
                    FewShotDt(c.dt.train.dt, **c.dt.few_shot.ini),
                    priority=-2
                    )

dt.few_shot.epoch_len = IL(lambda c:
                           len(c.dt.few_shot.dt) // c.loader.train.batch_size,
                           priority=-1)

# * 只开启L2损失。
# cfg.loss.loss_items.cam_lb.ini.loss_type = 'l2'
cfg.loss.loss_items.multi_cls.weights = 0.

# * 开启AMP。
cfg.model.ini.fp32 = False  # CLIP参数为FP16，确保准确的特征编码。
cfg.amp.enabled = True  # 使用AMP确保正确梯度。
