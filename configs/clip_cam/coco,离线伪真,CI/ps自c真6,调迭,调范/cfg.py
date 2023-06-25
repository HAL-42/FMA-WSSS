#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/4/6 21:19
@File    : base.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Cfg2Tune, Param2Tune

cfg = config = Cfg2Tune('configs/patterns/seg_metrics/cls_m_IoU,pra.py', 
                        cfgs_update_at_parser=('configs/clip_cam/coco,离线伪真,CI/base.py',))

cfg.rslt_dir = ...

# * 设置伪真值。
cfg.dt.train.ini.ps_mask_dir = 'experiment/clip_cam/coco,cl_loss/infer_bests/scale,真,224,1·67,20·5k迭,6150val/' \
                               'att1次,·6阈,aff2次,at_cam/ann=rsw3/seed'

# * 设置随机初始种子。
cfg.model.initialize_seed = Param2Tune(['真', '妙'])

# * 设置增强方式。
cfg.auger.train.ini.scale_crop_method.crop_size = Param2Tune([224, 272])
cfg.auger.train.ini.scale_crop_method.high_low_ratio = 1.67

# * 设置迭代次数。
cfg.solver.max_iter = Param2Tune([25000, 50000, 75000])
