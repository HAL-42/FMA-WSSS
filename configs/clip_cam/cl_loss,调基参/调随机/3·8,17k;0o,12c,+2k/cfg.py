#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/3 18:02
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Cfg2Tune, Param2Tune

cfg = config = Cfg2Tune('configs/patterns/seg_metrics/cls_m_IoU,pra.py')

cfg.rslt_dir = ...

# * 使用两种调率策略。
cfg._cfgs_update_at_parser = Param2Tune([('configs/clip_cam/cl_loss,调基参/base.py',) +
                                         c for c in
                                         [(),
                                          ('configs/clip_cam/cl_loss,调基参/_patches/3.8k,1e-4小lr.py',)]],
                                        optional_value_names=['17k_iter', '3·8k_iter'])

# * 设置不同的全局随机。
cfg.rand_seed = Param2Tune(list(range(2000, 14000, 2000)))
cfg.rand_ref.empty_leaf()  # 关闭随机参考（用一棵空子树替代）。
