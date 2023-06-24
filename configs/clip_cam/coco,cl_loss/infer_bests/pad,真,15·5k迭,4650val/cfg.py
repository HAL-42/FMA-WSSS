#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/6/24 5:01
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/infer_voc/base_coco.py')

cfg.model.resume_file = 'experiment/clip_cam/coco,cl_loss/pad/2,3,4轮,15std/' \
                        'initialize_seed=真,max_iter=15500/checkpoints/iter-4650.pth'
