#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/18 22:32
@File    : base.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

from libs.data import VOCAug2
from libs.seed_voter.max_iou_imp2 import MaxIoU_IMP2

cfg = config = Config()

cfg.rslt_dir = ...
cfg.rand_seed = 0

# * 配置数据集。
cfg.dt.ini.root = 'datasets'
cfg.dt.ini.split = 'val'
cfg.dt.ini.cls_labels_type = 'seg_cls_labels'
cfg.dt.ini.ps_mask_dir = None
cfg.dt.ini.rgb_img = True
cfg.dt.cls = VOCAug2

# * 配置种子点路径。
cfg.seed.dir = ...

# * 配置SAM标注路径。
cfg.sam_anns.dir = ...

# * 配置种子点投票器参数。
cfg.voter.ini.sam_seg_occupied_by_fg_thresh = 0.5
cfg.voter.ini.fg_occupied_by_sam_seg_thresh = 0.85
cfg.voter.ini.use_seed_when_no_sam = True

cfg.voter.cls = MaxIoU_IMP2

# * 保存与可视化。
cfg.viz.enable = True
cfg.viz.step = 100
