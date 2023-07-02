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
                      'configs/anns_seed/_patches/mm_prob.py')

cfg.rslt_dir = ...

cfg.dt.ini.split = 'val'

# * 配置CAM路径。
cfg.cam.dir = 'experiment/others/mmseg/m2f-sb22-bt4-40k-512x-gt/infer/best,ss/seg_preds'

# * 配置SAM标注路径。
cfg.sam_anns.dir = 'experiment/sam_auto_seg/vh,val/pattern_key=l2_nmsf_s1_rsw3/anns'

# * 配置种子生成参数。
cfg.seed.bg_methods = [{'method': 'alpha_bg', 'alpha': .9},
                       {'method': 'alpha_bg', 'alpha': 1},
                       {'method': 'alpha_bg', 'alpha': 2}]
