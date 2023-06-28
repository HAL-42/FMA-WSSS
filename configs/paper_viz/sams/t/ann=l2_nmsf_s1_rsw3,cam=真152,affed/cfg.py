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

cfg.rslt_dir = ...

cfg.dt.ini.split = 'train'

# * 配置CAM路径。
cfg.cam.dir = 'experiment/clip_cam/cl_loss,调基参/调上下文/infer_bests/s真,·015,2轮/aff2次,at_score,att2aff1次/cam_affed'

# * 配置SAM标注路径。
cfg.sam_anns.dir = 'experiment/sam_auto_seg/vh,t/pattern_key=l2_nmsf_s1_rsw3/anns'


cfg.seed.norm_firsts = [True]
cfg.seed.bg_methods = [{'method': 'pow', 'pow': .5}]

# * 修改可视化配置。
cfg.viz.step = 1
