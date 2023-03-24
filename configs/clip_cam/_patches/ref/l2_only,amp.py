#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/22 12:30
@File    : l2_only,amp.py
@Software: PyCharm
@Desc    : 
"""
import os.path as osp

from alchemy_cat.py_tools import Config, IL

cfg = config = Config()

# * 设定随机参考。
cfg.rand_seed = 0
cfg.rand_ref.ref_dir = 'experiment/clip_cam/调GCAM损/l2_only,amp'
cfg.rand_ref.rand_copy = IL(lambda c:
                            {'initial context': (osp.join(c.rand_ref.ref_dir, 'checkpoints/start.pth'),
                                                 osp.join(c.rslt_dir, 'checkpoints/start.pth'))})
