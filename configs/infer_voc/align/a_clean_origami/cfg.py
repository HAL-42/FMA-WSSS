#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/13 19:33
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/infer_voc/align/base.py')

cfg.rslt_dir = ...

cfg.model.ini.ctx_cfg.ctx_init = 'a clean origami'
