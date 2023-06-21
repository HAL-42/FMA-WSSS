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

from alchemy_cat.py_tools import Config, IL

cfg = config = Config('configs/sam_auto_seg/base.py',
                      'configs/sam_auto_seg/patterns/coco.py')

# * 在train上推理。
cfg.dt.ini.split_idx_num = (0, 8)
cfg.dt.ini.split = 'train'

# * 选择模型参数。
cfg.mask_gen.pattern_key = 'l2_nmsf_s1_rsw3'
cfg.mask_gen.ini = IL(lambda c: mask_gen_ini_cfgs[c.mask_gen.pattern_key].branch_copy())
