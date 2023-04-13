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
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/clip_cam/调cls/cl_loss/cfg.py')

cfg.loader.train.sub_iter_num = 2
