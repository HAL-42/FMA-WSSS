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
from functools import partial
import os.path as osp

from alchemy_cat.py_tools import Config, IL

from libs.data import VOCAug2
from libs.seeding.seed_argmax import seed_argmax_cuda
from libs.seeding.aff.caller import att_cam

cfg = config = Config()

cfg.rslt_dir = ...

# * 设定数据集。
# ** 设定验证集。
cfg.dt.val.ini.cls_labels_type = 'seg_cls_labels'
cfg.dt.val.ini.split = 'train_aug'
cfg.dt.val.dt = IL(lambda c:
                   VOCAug2(root='datasets', **c.dt.val.ini),
                   priority=-1)

# * 设定CAM优化。
cfg.aff.ori_cam_dir = IL(lambda c: osp.join(c.rslt_dir, '..', 'cam'))  # 默认在rslt_dir的上一级目录下寻找cam。

cfg.aff.ini.att2aff_cfg.last_n_layers = 8
cfg.aff.ini.att2aff_cfg.method.type = 'sink-horn'
cfg.aff.ini.att2aff_cfg.method.n_iter = 3  # 可调
cfg.aff.ini.aff_mask_cfg.method.type = 'thresh-bbox'
cfg.aff.ini.aff_mask_cfg.method.thresh = 0.4  # 可调
cfg.aff.ini.aff_mask_cfg.method.to_in_bbox = 'in_bbox'  # 可调
cfg.aff.ini.aff_mask_cfg.method.to_out_bbox = 'in_bbox'  # 可调
cfg.aff.ini.aff_cfg.n_iters = 1  # 可调
cfg.aff.ini.aff_at = 'cam'  # 试试score
cfg.aff.ini.dsize = None
cfg.aff.cal = IL(lambda c: partial(att_cam, **c.aff.ini))

# * 设定保存的内容。
cfg.solver.save_cam = True
cfg.solver.viz_cam = True
cfg.solver.viz_score.resize_first = IL(lambda c: c.eval.seed.ini.resize_first)
cfg.solver.viz_step = 100

# * 设定eval方法。
cfg.eval.enabled = True
cfg.eval.seed.cal = seed_argmax_cuda
cfg.eval.seed.bg_methods = [{'method': 'pow', 'pow': p} for p in range(1, 4)]
cfg.eval.seed.ini.resize_first = True  # 先阈值+归一化，还是先resize。
cfg.eval.seed.crf = None  # 可调
cfg.eval.seed.save = None
cfg.eval.seed.mask = None
