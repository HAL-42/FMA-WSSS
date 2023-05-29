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
import os.path as osp
from functools import partial

from alchemy_cat.py_tools import Cfg2Tune, Param2Tune, IL

cfg = config = Cfg2Tune('configs/patterns/seg_metrics/cls_m_IoU,pra.py',
                        cfgs_update_at_parser=('configs/clip_cam/l1_only,调基参/base.py',)
                        )

cfg.rslt_dir = ...

cfg.rand_ref.ref_dir = 'pretrains/rand_ref'
pre_ini_names = ('M=16/V1', 'M=16/V2k', 'M=16/V4k',
                 'M=16/V6k', 'M=16/V8k', 'M=16/V10k',
                 'seed=1,M=16,csc=False,cls_token_pos=end',
                 'seed=2,M=16,csc=False,cls_token_pos=end',
                 'seed=3,M=16,csc=False,cls_token_pos=end',
                 'seed=5,M=16,csc=False,cls_token_pos=end',
                 'seed=8,M=16,csc=False,cls_token_pos=end',
                 'seed=13,M=16,csc=False,cls_token_pos=end',
                 'hash_seed/seed=随,M=16,csc=False,cls_token_pos=end',
                 'hash_seed/seed=机,M=16,csc=False,cls_token_pos=end',
                 'hash_seed/seed=性,M=16,csc=False,cls_token_pos=end',
                 'hash_seed/seed=真,M=16,csc=False,cls_token_pos=end',
                 'hash_seed/seed=奇,M=16,csc=False,cls_token_pos=end',
                 'hash_seed/seed=妙,M=16,csc=False,cls_token_pos=end',
                 'hash_seed/std=·125/seed=随,M=16,csc=False,cls_token_pos=end',
                 'hash_seed/std=·125/seed=机,M=16,csc=False,cls_token_pos=end',
                 'hash_seed/std=·125/seed=性,M=16,csc=False,cls_token_pos=end',
                 'hash_seed/std=·125/seed=真,M=16,csc=False,cls_token_pos=end',
                 'hash_seed/std=·125/seed=奇,M=16,csc=False,cls_token_pos=end',
                 'hash_seed/std=·125/seed=妙,M=16,csc=False,cls_token_pos=end')
def rand_copy(c, name):  # noqa
    return {'initial context': (osp.join(c.rand_ref.ref_dir, f'coop_ctx/{name}.pth'),
                                osp.join(c.rslt_dir, 'checkpoints/start.pth'))}
cfg.rand_ref.rand_copy = Param2Tune([IL(partial(rand_copy, name=name)) for name in pre_ini_names],   # noqa
                                    optional_value_names=pre_ini_names)
