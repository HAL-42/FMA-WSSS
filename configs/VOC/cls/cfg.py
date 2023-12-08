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
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/patterns/seg_metrics/cls_m_IoU,pra.py',
                      cfgs_update_at_parser=('configs/clip_cam/cl_loss,调基参/base.py',))

cfg.rslt_dir = ...

# * 设置不同的随机初始种子。
cfg.rand_ref.empty_leaf()  # 关闭随机参考（用一棵空子树替代）。
cfg.model.initialize_seed = '真'

# * 设置模型初始化。
cfg.model.ini.ctx_cfg.ctx_std = 0.015

# * 设置不同的迭代次数。
cfg.solver.max_iter = 500 + 1300
