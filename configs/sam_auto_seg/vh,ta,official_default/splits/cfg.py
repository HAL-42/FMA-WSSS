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
from configs.sam_auto_seg.patterns.mask_gen_ini import cfg as mask_gen_ini_cfgs

from alchemy_cat.py_tools import Cfg2Tune, PL, Param2Tune

from libs.data import VOCAug2

cfg = config = Cfg2Tune(cfgs_update_at_parser=('configs/sam_auto_seg/base.py',))

cfg.rslt_dir = ...

# * 配置数据集。
cfg.dt.ini.root = 'datasets'
cfg.dt.ini.split = 'train_aug'
cfg.dt.ini.cls_labels_type = 'seg_cls_labels'
cfg.dt.ini.ps_mask_dir = None
cfg.dt.ini.rgb_img = True
cfg.dt.ini.split_idx_num = Param2Tune([(0, 4), (1, 4), (2, 4), (3, 4)])
cfg.dt.cls = VOCAug2.subset

# * 选择模型参数。
cfg.mask_gen.pattern_key = 'official_default'

cfg.mask_gen.ini = PL(lambda c: mask_gen_ini_cfgs[c.mask_gen.pattern_key].branch_copy())
