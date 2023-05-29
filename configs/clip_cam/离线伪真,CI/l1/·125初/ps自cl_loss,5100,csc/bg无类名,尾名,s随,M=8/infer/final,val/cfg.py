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

cfg.model.ini.ctx_cfg.n_ctx = 8
cfg.model.ini.ctx_cfg.csc = True

cfg.model.bg_names = [''] * 25

cfg.model.resume_file = 'experiment/clip_cam/离线伪真,CI/l1/·125初/ps自cl_loss,5100,csc/bg无类名,尾名,s随,M=8/' \
                        'checkpoints/final.pth'

cfg.dt.val.ini.split = 'val'
