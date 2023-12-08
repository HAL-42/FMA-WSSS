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

from configs.sam_auto_seg.patterns.mask_gen_ini import cfg as mask_gen_ini_cfgs

cfg = config = Cfg2Tune(cfgs_update_at_parser=('configs/sam_auto_seg/base.py',))

cfg.rslt_dir = ...

# * 设置训练、验证、测试集。
cfg.dt.ini.split = Param2Tune(['train_aug', 'val', 'test'])

# * 选择模型参数。
cfg.mask_gen.pattern_key = 'l2_nmsf_s1_rsw3'
@cfg.mask_gen.set_IL()  # noqa: E302
def ini(c):
    return mask_gen_ini_cfgs[c.mask_gen.pattern_key].branch_copy()
