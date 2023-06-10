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

cfg = config = Config('configs/anns_seed/base.py',
                      'configs/anns_seed/_patches/ann_prob.py',
                      'configs/anns_seed/_patches/dp_logit.py')

cfg.rslt_dir = ...

cfg.dt.ini.split = 'val'

# * 配置CAM路径。
cfg.cam.dir = 'experiment/others/deeplab-pytorch/features/无背头6,crf/deeplabv2_resnet101_msc/val/logit'

# * 配置SAM标注路径。
cfg.sam_anns.dir = 'experiment/sam_auto_seg/vh,val/pattern_key=l2_nmsf_s1_rsw3/anns'

# * 配置cam加载器。
cfg.cam.loader.ini.must_in_seed = True
