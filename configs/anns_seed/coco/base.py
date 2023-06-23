#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/18 22:32
@File    : base.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config, IL

from libs.data import COCO

cfg = config = Config('configs/anns_seed/base.py')

cfg.rslt_dir = ...
cfg.rand_seed = 0

# * 配置数据集。
cfg.dt.ini.root = 'datasets'
cfg.dt.ini.split = 'train'
cfg.dt.ini.ps_mask_dir = None
cfg.dt.ini.rgb_img = True
cfg.dt.cls = COCO

# * 配置SAM标注路径。
cfg.sam_anns.dir = IL(lambda c: f'experiment/sam_auto_seg/co,{c.dt.ini.split[0]}/all/anns')

# * 配置种子生成参数。
cfg.seed.norm_firsts = [True]
cfg.seed.bg_methods = [{'method': 'pow', 'pow': .5},
                       {'method': 'pow', 'pow': .6},
                       {'method': 'pow', 'pow': .7},
                       {'method': 'pow', 'pow': .8},
                       {'method': 'pow', 'pow': .9},
                       {'method': 'pow', 'pow': 1},
                       {'method': 'pow', 'pow': 2},
                       {'method': 'pow', 'pow': 3}]

# * 保存与可视化。
cfg.viz.step = 1000
