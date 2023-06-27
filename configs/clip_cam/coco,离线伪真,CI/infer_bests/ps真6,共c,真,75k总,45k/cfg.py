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

cfg.model.resume_file = 'experiment/clip_cam/coco,离线伪真,CI/ps自c真6,共ctx,调迭,调范/' \
                        'initialize_seed=真,max_iter=75000/checkpoints/iter-45000.pth'

cfg.eval.seed.bg_methods = [{'method': 'pow', 'pow': p} for p in range(4, 7)]
