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

cfg.model.resume_file = 'experiment/clip_cam/cl_loss,调基参/(·2,1c,·2)小lr,(5,20c,5)轮/二轮/(·2,1c,·2)小lr,7·5轮/' \
                        'eta_min=0·0002/checkpoints/iter-3000.pth'
