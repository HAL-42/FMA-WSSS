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

# * 在val上推理。
cfg.dt.ini.split = 'val'

# * 选择模型参数。
cfg.mask_gen.pattern_key = Param2Tune(['l2_only_s1',
                                       'l2_nmsf_s1_rsw3',
                                       'ssa_default',
                                       'l2_nmsf_s1',
                                       'l2_only_s1_t4',
                                       'l2_nmsf_s1_c0',
                                       'l2_nmsf',
                                       'l2_only',
                                       'ssa_light',
                                       'ssa_light_8p',
                                       'official_heavy',
                                       'ssa_heavy',
                                       'official_default',
                                       ])
cfg.mask_gen.ini = PL(lambda c: mask_gen_ini_cfgs[c.mask_gen.pattern_key].branch_copy())
