#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/12 10:26
@File    : s真,M=16.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/clip_cam/_patches/ref/coop_ctx-·125初/base.py')

cfg.rand_ref.ini_rand_base = '真'

cfg.model.ini.ctx_cfg.n_ctx = 16
