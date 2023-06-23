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

cfg = config = Config('configs/infer_voc/patch_val_coco.py')  # 基于coco val配置。

# * 设定eval方法。
cfg.eval.seed.bg_methods = [{'method': 'pow', 'pow': p} for p in range(1, 11)]
