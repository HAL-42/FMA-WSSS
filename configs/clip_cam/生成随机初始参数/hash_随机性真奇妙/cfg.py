#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/7 11:14
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Cfg2Tune, Param2Tune, ParamLazy

cfg = config = Cfg2Tune(cfgs_update_at_parser=('configs/clip_cam/base.py',))

cfg.rslt_dir = ...

# * 使用不同随机种子。
seeds = '随机性真奇妙'
cfg.seed_idx = Param2Tune(list(range(6)))
cfg.rand_seed = ParamLazy(lambda c: hash(seeds[c.seed_idx]))
cfg.rand_base = ParamLazy(lambda c: seeds[c.seed_idx])

# * 使用不同的模型结构。
cfg.model.ini.ctx_cfg.n_ctx = Param2Tune([4, 8, 16, 24, 32, 48, 64])
cfg.model.ini.ctx_cfg.csc = Param2Tune([False, True])
cfg.model.ini.ctx_cfg.cls_token_pos = Param2Tune(['front', 'middle', 'end'])
