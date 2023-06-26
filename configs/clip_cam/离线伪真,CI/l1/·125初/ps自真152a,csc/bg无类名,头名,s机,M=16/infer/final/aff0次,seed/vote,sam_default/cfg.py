#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/3 15:41
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/seed_vote/base.py',)

# * 在val上推理。
cfg.dt.ini.split = 'train_aug'

# * 指定种子和标注位置。
cfg.seed.dir = 'experiment/clip_cam/离线伪真,CI/l1/·125初/ps自真152a,csc/bg无类名,头名,s机,M=16/infer/final/aff0次,seed' \
               '/seed/best/mask'

cfg.sam_anns.dir = 'experiment/sam_auto_seg/vh,ta/pattern_key=l2_nmsf_s1_rsw3/anns'

# * 选择模型参数。
cfg.voter.ini.sam_seg_occupied_by_fg_thresh = .4
cfg.voter.ini.fg_occupied_by_sam_seg_thresh = 0.65
cfg.voter.ini.use_seed_when_no_sam = True
