#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/19 22:21
@File    : patch_train.py
@Software: PyCharm
@Desc    : 若用于训练时验证，打的补丁。
"""
from alchemy_cat.py_tools import Config

cfg = config = Config()

# * 不保存除eval外的任何中间结果。
cfg.solver.save_att = 0
cfg.solver.save_cam = False
cfg.solver.viz_cam = False
cfg.solver.viz_score = False

# * eval减少搜索空间。
cfg.eval.seed.bg_methods = [{'method': 'pow', 'pow': p} for p in (5, 10)]  # 扩大网眼，两点足以确定斜率已知的调参抛物线。
