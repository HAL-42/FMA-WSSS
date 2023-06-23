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
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/anns_seed/coco/base.py',
                      'configs/anns_seed/_patches/ann_prob.py',
                      'configs/anns_seed/_patches/mm_prob.py')

cfg.dt.ini.split = 'val'

# * 配置CAM路径。
cfg.cam.dir = 'experiment/others/mmseg/m2f-sl22-bt4-80k-512x-ES-50q/infer/75k,ss/seg_preds'
