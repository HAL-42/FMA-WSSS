#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/4/13 23:19
@File    : base.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/clip_cam/离线伪真,CI/l1/base.py')

# * 目前没有匹配的随机参考。
cfg.rand_ref.empty_leaf()

# * 使用cl_loss，5100 val时的伪真值作为监督。
cfg.dt.train.ini.ps_mask_dir = 'experiment/clip_cam/调cls/cl_loss/infer/5100/aff2次,at_score,ce_npp_crf,' \
                               'mask/seed/best/mask'

cfg.model.ini.clip_name = 'ViT-L/14@336px'

cfg.auger.train.ini.scale_crop_method.method = 'rand_range'
cfg.auger.train.ini.scale_crop_method.low_size = 470
cfg.auger.train.ini.scale_crop_method.high_size = 706
cfg.auger.train.ini.scale_crop_method.short_thresh = 336  # 希望短边从336~(336 * 1.5)，长短比≈1.4，算出长边范围。
cfg.auger.train.ini.scale_crop_method.crop_size = 336

cfg.loader.train.sub_iter_num = 4
