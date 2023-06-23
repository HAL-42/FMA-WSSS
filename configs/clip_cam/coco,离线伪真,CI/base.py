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
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/clip_cam/coco_base.py',
                      'configs/clip_cam/_patches/l1_only.py')  # 关闭分类损失。

# * 设置伪真值。
cfg.dt.train.ini.ps_mask_dir = ...

# * 设置增强方式。
cfg.auger.train.ini.scale_crop_method.crop_size = ...
cfg.auger.train.ini.scale_crop_method.high_low_ratio = ...

# * 设置随机初始种子。
cfg.model.initialize_seed = ...

# * 使用分割Prompt设置。
cfg.model.ini.ctx_cfg.n_ctx = 16
cfg.model.ini.ctx_cfg.csc = True
cfg.model.ini.ctx_cfg.cls_token_pos = 'front'

cfg.model.bg_names = [''] * 25

# * 设置迭代次数。
cfg.solver.max_iter = ...
