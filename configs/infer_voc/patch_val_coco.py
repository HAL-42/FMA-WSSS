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

cfg = config = Config('configs/infer_voc/patch_val.py')

# * 设定数据集。
cfg.dt.val.ini.subsplit = '1250'
