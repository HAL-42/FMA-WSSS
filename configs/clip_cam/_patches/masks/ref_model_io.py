#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/25 22:09
@File    : ref_model_io.py
@Software: PyCharm
@Desc    : Mask出模型和IO，共infer继承。
"""
from alchemy_cat.py_tools import Config

cfg = config = Config()

cfg.rslt_dir = ...
cfg.rand_seed = 0

# cfg.rand_ref.empty_leaf()

cfg.dt.empty_leaf()

cfg.auger.empty_leaf()

cfg.loader.empty_leaf()

# cfg.io.empty_leaf()

# cfg.model.empty_leaf()

cfg.opt.empty_leaf()

cfg.sched.empty_leaf()

cfg.loss.empty_leaf()

cfg.amp.empty_leaf()

cfg.solver.empty_leaf()

cfg.val.empty_leaf()

cfg.infer.empty_leaf()
