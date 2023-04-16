#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/4/13 11:47
@File    : 17k调率,5·1k截.py
@Software: PyCharm
@Desc    : 模拟此前次优的val曲线。
"""
from alchemy_cat.py_tools import Config

cfg = config = Config()

cfg.rslt_dir = ...

# * 最小学习率是初始学习率的0.1。
cfg.sched.main.ini.eta_min = 0.001 * 0.1

# * 一律预热500迭代（0.75）轮，训练5,10,15,20轮。
cfg.solver.max_iter = 500 + 3300
