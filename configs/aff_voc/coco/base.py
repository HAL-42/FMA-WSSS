#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/10 21:27
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config, IL

from libs.data import COCO

cfg = config = Config(cfgs_update_at_parser=('configs/aff_voc/base.py',))

# * 设定数据集。
cfg.dt.val.set_whole(True)
cfg.dt.val.dt = IL(lambda c:
                   COCO(root='datasets', split='train', **c.dt.val.ini),
                   priority=-1)

# * 考虑到CAM性能较差，使用官方参数（而非VOC精调版），并将thresh提高到0.7。
cfg.aff.ini.att2aff_cfg.method.n_iter = 3
cfg.aff.ini.aff_mask_cfg.method.thresh = 0.7
cfg.aff.ini.aff_cfg.n_iters = 1
cfg.aff.ini.aff_at = 'score'

# * 设定保存的内容。
cfg.solver.viz_step = 1000

# * 设定eval方法。
cfg.eval.seed.bg_methods = [{'method': 'pow', 'pow': p} for p in range(1, 4)]
