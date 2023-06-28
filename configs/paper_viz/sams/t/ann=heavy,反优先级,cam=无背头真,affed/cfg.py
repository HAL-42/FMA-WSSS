#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/25 16:05
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/anns_seed/base.py')

cfg.dt.ini.split = 'train'

cfg.sam_anns.dir = 'experiment/sam_auto_seg/vh,t/pattern_key=ssa_heavy/anns'

# * 配置CAM路径。
cfg.cam.dir = 'experiment/clip_cam/离线伪真,CI/l1/·125初/ps自真152a,csc/bg无类名,头名,s机,M=16/infer/final,train/' \
              'aff2次,at_cam,att1次,·5掩阈/cam_affed'

# * 修改得分算法参数。
cfg.seed.norm_firsts = [True]
cfg.seed.bg_methods = [{'method': 'pow', 'pow': .5}]

cfg.seed.ini.priority = ('level_smaller',)

# * 修改可视化配置。
cfg.viz.step = 1
