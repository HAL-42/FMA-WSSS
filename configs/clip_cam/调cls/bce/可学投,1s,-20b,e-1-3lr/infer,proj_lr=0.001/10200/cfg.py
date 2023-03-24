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

cfg.model.resume_file = 'experiment/clip_cam/调cls/bce/可学投,1s,-20b,e-1-3lr/proj_lr=0·001/checkpoints/iter-10200.pth'
