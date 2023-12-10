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

cfg.model.resume_file = 'experiment/COCO/seg/checkpoints/iter-45000.pth'

cfg.eval.seed.bg_methods = [{'method': 'pow', 'pow': p} for p in range(4, 7)]
