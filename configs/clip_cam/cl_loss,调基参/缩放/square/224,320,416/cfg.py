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
from alchemy_cat.py_tools import Cfg2Tune, Param2Tune, PL

cfg = config = Cfg2Tune('configs/patterns/seg_metrics/cls_m_IoU,pra.py')

cfg.rslt_dir = ...

# * 实验3800和17000两种迭代数。
cfg._cfgs_update_at_parser = Param2Tune([('configs/clip_cam/cl_loss,调基参/base.py',) + c
                                         for c in
                                         [(),
                                          ('configs/clip_cam/cl_loss,调基参/_patches/3.8k,1e-4小lr.py',)]],
                                        optional_value_names=['17k_iter', '3·8k_iter'])

# * 实验不同scale策略。
cfg.auger.train.ini.scale_crop_method = Param2Tune([224, 320, 416, 512])

# * sub iter数目提高到4，防止显存不足。
cfg.loader.train.sub_iter_num = PL(lambda c: {224: 1, 320: 2, 416: 4, 512: 8}[c.auger.train.ini.scale_crop_method])
