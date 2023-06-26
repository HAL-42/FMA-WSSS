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

# * 使用分割Prompt设置。
cfg.model.ini.ctx_cfg.n_ctx = 16
cfg.model.ini.ctx_cfg.csc = False
cfg.model.ini.ctx_cfg.cls_token_pos = 'end'

cfg.model.bg_names = ['ground', 'land', 'grass', 'tree', 'building', 'wall', 'sky', 'lake', 'water', 'river', 'sea',
                      'railway', 'railroad', 'helmet',
                      'cloud', 'house', 'mountain', 'ocean', 'road', 'rock', 'street', 'valley', 'bridge',
                      ]  # 相比VOC，去掉了keyboard和sign。

# * 设置不同的随机初始种子。
cfg.model.initialize_seed = Param2Tune(['真'])

# * 设置模型初始化。
cfg.model.ini.ctx_cfg.ctx_std = 0.015

# * 设置增强方式。
cfg.auger.train.ini.scale_crop_method.crop_size = 224
cfg.auger.train.ini.scale_crop_method.high_low_ratio = 1.67

cfg.loader.train.sub_iter_num = 8

# * 设置迭代次数。
cfg.solver.max_iter = Param2Tune([25000, 50000, 75000])
