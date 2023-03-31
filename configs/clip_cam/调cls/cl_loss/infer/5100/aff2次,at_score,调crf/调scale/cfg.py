#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/29 16:14
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Cfg2Tune, Param2Tune

cfg = config = Cfg2Tune('configs/patterns/seg_metrics/cls_m_IoU,pra.py',
                        cfgs_update_at_parser=('configs/clip_cam/调cls/cl_loss/infer/5100/aff2次,at_score,调crf/base.py',
                                               'configs/patterns/crf/scale_crf.py'))

cfg.rslt_dir = ...

cfg.crf.scale = Param2Tune([1, 6, 12])
