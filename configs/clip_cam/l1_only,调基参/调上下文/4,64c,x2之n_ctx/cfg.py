#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/3 15:41
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Cfg2Tune, Param2Tune

cfg = config = Cfg2Tune('configs/patterns/seg_metrics/cls_m_IoU,pra.py',
                        cfgs_update_at_parser=('configs/clip_cam/l1_only,调基参/base.py',))

cfg.rslt_dir = ...

# * 使用不同随机种子。
cfg.rand_seed = Param2Tune(list(range(0, 10001, 2000)))

# * 实验长度为4、8、16、32、64的上下文。
cfg.model.ini.ctx_cfg.n_ctx = Param2Tune([4, 8, 16, 32, 64])
