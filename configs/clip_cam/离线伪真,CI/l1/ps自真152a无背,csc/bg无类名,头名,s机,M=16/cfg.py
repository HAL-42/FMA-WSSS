#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/4/13 23:19
@File    : base.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/clip_cam/离线伪真,CI/l1/base.py',
                      'configs/clip_cam/_patches/ref/coop_ctx-·125初/base.py')

# * 使用cl_loss，5100 val时的伪真值作为监督。
cfg.dt.train.ini.ps_mask_dir = 'experiment/anns_seed/ta/ann=l2_nmsf_s1_rsw3,cam=真152无背,affed/seed'

# * 使用不同的随机基元，对应不同的初始ctx（数据流不变）。
cfg.rand_ref.ini_rand_base = '机'

# * 实验长度为8、16、32的上下文。
cfg.model.ini.ctx_cfg.n_ctx = 16
cfg.model.ini.ctx_cfg.csc = True
cfg.model.ini.ctx_cfg.cls_token_pos = 'front'

cfg.model.bg_names = [''] * 25
