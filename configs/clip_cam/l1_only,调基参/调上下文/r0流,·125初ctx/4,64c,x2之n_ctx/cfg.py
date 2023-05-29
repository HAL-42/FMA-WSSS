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
import os.path as osp

from alchemy_cat.py_tools import Cfg2Tune, Param2Tune, Config

cfg = config = Cfg2Tune('configs/patterns/seg_metrics/cls_m_IoU,pra.py',
                        cfgs_update_at_parser=('configs/clip_cam/l1_only,调基参/base.py',))

cfg.rslt_dir = ...

# * 使用不同的随机基元，对应不同的初始ctx（数据流不变）。
cfg.rand_base = Param2Tune(['随', '机', '性', '真', '奇', '妙'])

# * 实验长度为4、8、16、32、64的上下文。
cfg.model.ini.ctx_cfg.n_ctx = Param2Tune([4, 8, 16, 32, 64])

# * 使用不同的随机参考。
cfg.rand_ref.ref_dir = 'pretrains/rand_ref/coop_ctx/hash_seed/std=·125'

@cfg.rand_ref.set_IL()  # noqa
def rand_copy(c: Config):
    ctx_cfg = c.model.ini.ctx_cfg
    return {'initial context': (osp.join(c.rand_ref.ref_dir,
                                         f'seed={c.rand_base},'
                                         f'M={ctx_cfg.n_ctx},'
                                         f'csc={ctx_cfg.csc},'
                                         f'cls_token_pos={ctx_cfg.cls_token_pos}.pth'),
                                osp.join(c.rslt_dir, 'checkpoints/start.pth'))}
