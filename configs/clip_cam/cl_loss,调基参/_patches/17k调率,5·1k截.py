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

cfg.sched.main.ini.T_max = 17000 - 500

cfg.solver.max_iter = 5100
