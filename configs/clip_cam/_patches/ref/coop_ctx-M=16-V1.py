#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/22 12:30
@File    : coop_ctx-M=16-V1.py
@Software: PyCharm
@Desc    : 
"""
import os.path as osp

from alchemy_cat.py_tools import Config, IL

cfg = config = Config()

# * 设定随机参考。
cfg.rand_seed = 0
cfg.rand_ref.ref_dir = 'pretrains/rand_ref'
cfg.rand_ref.rand_copy = IL(lambda c:
                            {'initial context': (osp.join(c.rand_ref.ref_dir, 'coop_ctx/M=16/V1.pth'),
                                                 osp.join(c.rslt_dir, 'checkpoints/start.pth'))})
