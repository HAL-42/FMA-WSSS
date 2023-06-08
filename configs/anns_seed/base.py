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
from functools import partial

from alchemy_cat.py_tools import Config, IL

from libs.data import VOCAug2
from libs.seeding import seed_anns

cfg = config = Config()

cfg.rslt_dir = ...
cfg.rand_seed = 0

# * 配置数据集。
cfg.dt.ini.root = 'datasets'
cfg.dt.ini.split = 'val'
cfg.dt.ini.cls_labels_type = 'seg_cls_labels'
cfg.dt.ini.ps_mask_dir = None
cfg.dt.ini.rgb_img = True
cfg.dt.cls = VOCAug2

# * 配置CAM路径。
cfg.cam.dir = ...
cfg.cam.loader = None

# * 配置替补种子点路径。
cfg.seed.dir = ...

# * 配置SAM标注路径。
cfg.sam_anns.dir = ...

# * 配置种子生成参数。
cfg.seed.norm_firsts = [True, False]
cfg.seed.bg_methods = [{'method': 'pow', 'pow': .5},
                       {'method': 'pow', 'pow': .6},
                       {'method': 'pow', 'pow': .7},
                       {'method': 'pow', 'pow': .8},
                       {'method': 'pow', 'pow': .9},
                       {'method': 'pow', 'pow': 1},
                       {'method': 'pow', 'pow': 2},
                       {'method': 'pow', 'pow': 3}]

cfg.seed.ini.priority = ('level_bigger', 'conf_bigger')
cfg.seed.ini.gather_method = 'mean'
cfg.seed.cal = IL(lambda c: partial(seed_anns.gather_norm_bg_argmax, **c.seed.ini))

# * 保存与可视化。
cfg.viz.enable = True
cfg.viz.step = 100
