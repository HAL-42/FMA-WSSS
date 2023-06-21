#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/6/20 14:55
@File    : coco.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

from libs.data.coco_dt import COCO

cfg = config = Config()

cfg.dt.set_whole()
cfg.dt.ini.split_idx_num = (0, 1)
cfg.dt.ini.root = 'datasets'
cfg.dt.ini.split = 'train'
cfg.dt.ini.subsplit = ''
cfg.dt.ini.label_type = ''  # 无所谓。
cfg.dt.ini.cls_labels_type = 'seg_cls_labels'  # 无所谓。
cfg.dt.ini.ps_mask_dir = None
cfg.dt.ini.rgb_img = True

cfg.dt.cls = COCO.subset

cfg.viz.step = 1000
