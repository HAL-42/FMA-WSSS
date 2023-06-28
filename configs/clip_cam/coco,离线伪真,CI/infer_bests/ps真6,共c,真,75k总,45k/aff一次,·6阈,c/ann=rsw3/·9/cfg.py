#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/25 16:05
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
import os.path as osp

from alchemy_cat.py_tools import Config, IL

cfg = config = Config('configs/anns_seed/coco/base.py')

cfg.dt.ini.split = 'train'

# * 配置CAM路径。
cfg.cam.dir = IL(lambda c: osp.join(c.rslt_dir, '../..', 'cam_affed'))

cfg.seed.bg_methods = [{'method': 'pow', 'pow': .9}]

cfg.viz.enable = False
