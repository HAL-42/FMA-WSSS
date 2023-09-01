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
cfg.cam.dir = 'experiment/others/mmseg/d3p-r1d8-bt16-160e-512x-CO_s真-2au/infer/best,ss/seg_preds'

# * 配置超参。
cfg.seed.bg_methods = [{'method': 'alpha_bg', 'alpha': .6},
                       {'method': 'alpha_bg', 'alpha': .7},
                       {'method': 'alpha_bg', 'alpha': .8},
                       {'method': 'alpha_bg', 'alpha': .9},
                       {'method': 'alpha_bg', 'alpha': 1}]

