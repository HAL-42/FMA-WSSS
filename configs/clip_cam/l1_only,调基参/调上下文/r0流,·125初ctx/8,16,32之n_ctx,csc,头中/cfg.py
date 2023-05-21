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

# * 实验长度为4、8、16、32、64的上下文。
cfg.model.ini.ctx_cfg.n_ctx = Param2Tune([8, 16, 32])
cfg.model.ini.ctx_cfg.csc = Param2Tune([True])
cfg.model.ini.ctx_cfg.cls_token_pos = Param2Tune(['front', 'middle'])
