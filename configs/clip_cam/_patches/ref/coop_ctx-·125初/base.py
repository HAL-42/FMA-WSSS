#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/22 12:30
@File    : coop_ctx-M=16-V1.py
@Software: PyCharm
@Desc    : 
"""
import os.path as osp

from alchemy_cat.py_tools import Config

cfg = config = Config()

# * 设定随机参考。
cfg.rand_seed = 0

cfg.rand_ref.ref_dir = 'pretrains/rand_ref/coop_ctx/hash_seed/std=·125'
cfg.rand_ref.ini_rand_base = ...

@cfg.rand_ref.set_IL()  # noqa
def rand_copy(c: Config):
    ctx_cfg = c.model.ini.ctx_cfg
    return {'initial context': (osp.join(c.rand_ref.ref_dir,
                                         f'seed={c.rand_ref.ini_rand_base},'
                                         f'M={ctx_cfg.n_ctx},'
                                         f'csc={ctx_cfg.csc},'
                                         f'cls_token_pos={ctx_cfg.cls_token_pos}.pth'),
                                osp.join(c.rslt_dir, 'checkpoints/start.pth'))}
