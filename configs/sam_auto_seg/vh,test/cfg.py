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

from alchemy_cat.py_tools import Cfg2Tune, Param2Tune, PL

cfg = config = Cfg2Tune(cfgs_update_at_parser=('configs/sam_auto_seg/base.py',))

cfg.rslt_dir = ...

# * 在train上推理。
cfg.dt.ini.split = 'test'

# * 选择模型参数。
cfg.mask_gen.pattern_key = Param2Tune(['l2_nmsf_s1_rsw3'])
cfg.mask_gen.ini = PL(lambda c: mask_gen_ini_cfgs[c.mask_gen.pattern_key].branch_copy())
