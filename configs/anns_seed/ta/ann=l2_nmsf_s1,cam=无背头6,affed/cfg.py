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

cfg.dt.ini.split = 'train_aug'

# * 配置CAM路径。
cfg.cam.dir = 'experiment/clip_cam/离线伪真,CI/l1/·125初/ps自cl_loss,5100,csc/bg无类名,头名,s机,M=16/infer/final/' \
              'aff2次,at_cam,att1次,·5掩阈,ce_npp/cam_affed'

# * 配置替补种子点路径。
cfg.seed.dir = 'experiment/clip_cam/离线伪真,CI/l1/·125初/ps自cl_loss,5100,csc/bg无类名,头名,s机,M=16/infer/final/' \
               'aff2次,at_cam,att1次,·5掩阈,ce_npp/seed/best/mask'

# * 配置SAM标注路径。
cfg.sam_anns.dir = 'experiment/sam_auto_seg/vh,ta/pattern_key=l2_nmsf_s1/anns'

# * 配置种子生成参数。
cfg.seed.norm_firsts = [True]
