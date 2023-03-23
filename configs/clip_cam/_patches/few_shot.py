#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/22 11:59
@File    : patch_few_shot.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config, IL
from libs.data import FewShotDt

cfg = config = Config()

# * 定义小样本数据集。
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

# * 根据训练轮次，决定迭代和预热次数。
cfg.sched.warm.warm_epoch = ...
cfg.sched.warm.warm_iters = IL(lambda c:
                               round(c.dt.few_shot.epoch_len * c.sched.warm.warm_epoch),
                               priority=0)

cfg.solver.max_epoch = ...
cfg.solver.max_iter = IL(lambda c:
                         round(c.dt.few_shot.epoch_len * c.solver.max_epoch),
                         priority=0)
