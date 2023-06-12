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
                      'configs/anns_seed/_patches/wt_prob.py')

cfg.rslt_dir = ...

cfg.dt.ini.split = 'val'

# * 配置CAM路径。
cfg.cam.dir = 'experiment/WeakTr/ann=l2_nmsf_s1_rsw3,cam=无背头6,affed,no_clip,vit/seg_prob_ss'

# * 配置SAM标注路径。
cfg.sam_anns.dir = 'experiment/sam_auto_seg/vh,val/pattern_key=l2_nmsf_s1_rsw3/anns'
