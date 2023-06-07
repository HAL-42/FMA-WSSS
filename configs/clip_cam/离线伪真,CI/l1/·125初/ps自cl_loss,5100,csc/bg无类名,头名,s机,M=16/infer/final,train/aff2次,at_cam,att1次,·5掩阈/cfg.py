#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/29 16:14
@File    : cfg.py
@Software: PyCharm
@Desc    : 没有CRF情况下，寻找最优aff配置。
"""
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/aff_voc/base.py',)

cfg.rslt_dir = ...

cfg.dt.val.ini.split = 'train'

# * 修改算法参数。
cfg.aff.ini.att2aff_cfg.method.n_iter = 1
cfg.aff.ini.aff_mask_cfg.method.thresh = .5

cfg.aff.ini.aff_cfg.n_iters = 2
cfg.aff.ini.aff_at = 'cam'
