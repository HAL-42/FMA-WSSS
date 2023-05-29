#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/3 15:41
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Cfg2Tune, Param2Tune

cfg = config = Cfg2Tune('configs/patterns/seg_metrics/cls_m_IoU,pra.py',
                        cfgs_update_at_parser=('configs/clip_cam/l1_only,调基参/base.py',
                                               'configs/clip_cam/_patches/ref/coop_ctx-·125初/base.py'))

cfg.rslt_dir = ...

# * 使用不同的随机基元，对应不同的初始ctx（数据流不变）。
cfg.rand_ref.ini_rand_base = Param2Tune(['随', '机', '性', '真', '奇', '妙'])

# * 实验长度为8、16、32的上下文。
cfg.model.ini.ctx_cfg.n_ctx = Param2Tune([8, 16, 32])
cfg.model.ini.ctx_cfg.csc = True

# * 设置前背景类名。
fg_names = ['aeroplane', 'bicycle', 'bird avian', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair seat', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person with clothes,people,human',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor screen',
            ]
zero_fg_names = [''] * len(fg_names)

bg_names = ['ground', 'land', 'grass', 'tree', 'building',
            'wall', 'sky', 'lake', 'water', 'river',
            'sea', 'railway', 'railroad', 'keyboard', 'helmet',
            'cloud', 'house', 'mountain', 'ocean', 'road',
            'rock', 'street', 'valley', 'bridge', 'sign',
            ]
zero_bg_names = [''] * len(bg_names)

cfg.model.fg_names = Param2Tune([fg_names, zero_fg_names], optional_value_names=['clip_es_fg', 'zero_fg'])
cfg.model.bg_names = Param2Tune([bg_names, zero_bg_names], optional_value_names=['clip_es_bg', 'zero_bg'],
                                subject_to=lambda x: x[0] == '' or cfg.model.fg_names.cur_val[0] == '')
