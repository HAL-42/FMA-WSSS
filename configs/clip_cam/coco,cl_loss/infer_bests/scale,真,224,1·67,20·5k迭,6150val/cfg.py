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

cfg.model.resume_file = 'experiment/clip_cam/coco,cl_loss/scale/2,3,4轮,15std,调范/' \
                        'initialize_seed=真,crop_size=224,high_low_ratio=1.6666666666666667,max_iter=20500/' \
                        'checkpoints/iter-6150.pth'
