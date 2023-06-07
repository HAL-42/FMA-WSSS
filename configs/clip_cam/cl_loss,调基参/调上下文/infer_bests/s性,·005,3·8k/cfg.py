#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/13 20:39
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/infer_voc/align/base.py')

cfg.rslt_dir = ...

cfg.model.resume_file = 'experiment/clip_cam/cl_loss,调基参/调上下文/3·8,17k;515std/' \
                        '_cfgs_update_at_parser=3·8k_iter,initialize_seed=性,ctx_std=0.005/checkpoints/final.pth'
