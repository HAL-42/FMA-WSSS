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
cfg.dt.ini.split = 'train'

# * 指定种子和标注位置。
cfg.seed.dir = 'experiment/clip_cam/离线伪真,CI/l1/·125初/ps自真152a,csc/bg无类名,头名,s机,M=16/infer/' \
               'final,train/aff2次,at_cam,att1次,·5掩阈,seed/seed/best/mask'

cfg.sam_anns.dir = 'experiment/sam_auto_seg/vh,t/pattern_key=official_default/anns'

# * 选择模型参数。
cfg.voter.ini.sam_seg_occupied_by_fg_thresh = .5
cfg.voter.ini.fg_occupied_by_sam_seg_thresh = 0.75
cfg.voter.ini.use_seed_when_no_sam = True
