#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/7 11:14
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Cfg2Tune, Param2Tune

cfg = config = Cfg2Tune('configs/patterns/seg_metrics/cls_m_IoU,pra.py',
                        cfgs_update_at_parser=('configs/clip_cam/l1_only,调基参/base.py',
                                               'configs/clip_cam/_patches/ref/coop_ctx-M=16-V1.py')   # 固定初始ctx。
                        )

cfg.rslt_dir = ...

# * 使用不同随机种子。
cfg.rand_seed = Param2Tune(list(range(0, 10001, 2000)))
