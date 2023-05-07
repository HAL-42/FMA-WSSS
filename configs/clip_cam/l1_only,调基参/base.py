#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/4/6 21:19
@File    : base.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/clip_cam/base.py',
                      # 'configs/clip_cam/_patches/ref/coop_ctx-M=16-V1.py',  # n_ctx不确定。
                      'configs/clip_cam/_patches/l1_only.py')
