#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/10 12:16
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Cfg2Tune, Param2Tune

cfg = config = Cfg2Tune('configs/patterns/seg_metrics/cls_m_IoU,pra.py')

cfg.rslt_dir = ...

# * 实验3800和17000两种迭代数。
cfg._cfgs_update_at_parser = Param2Tune([('configs/clip_cam/cl_loss,调基参/base.py',
                                          'configs/patterns/aug/voc_rand_range.py',) + c
                                         for c in
                                         [(),
                                          ('configs/clip_cam/cl_loss,调基参/_patches/3.8k,1e-4小lr.py',)]])

# * 实验不同scale策略。
cfg.auger.train.ini.scale_crop_method.crop_size = Param2Tune([224, 320])
cfg.auger.train.min_area_propor = Param2Tune([0.3, 0.5, 0.7])

# * sub iter数目提高到2，防止显存不足。
cfg.loader.train.sub_iter_num = 2

# * 关掉infer的可视化，加快实验。
cfg.infer.cfg.solver.viz_cam = False
cfg.infer.cfg.solver.viz_score = False
