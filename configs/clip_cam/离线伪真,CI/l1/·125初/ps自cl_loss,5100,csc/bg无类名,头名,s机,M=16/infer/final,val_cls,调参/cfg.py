#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/13 20:39
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Cfg2Tune, IL, Param2Tune

cfg = config = Cfg2Tune('configs/patterns/seg_metrics/cls_m_IoU,pra.py',
                        cfgs_update_at_parser=('configs/infer_voc/align/base.py',))

cfg.rslt_dir = ...

cfg.model.ini.ctx_cfg.n_ctx = 16
cfg.model.ini.ctx_cfg.csc = True
cfg.model.ini.ctx_cfg.cls_token_pos = 'front'

cfg.model.bg_names = [''] * 25

cfg.model.resume_file = 'experiment/clip_cam/离线伪真,CI/l1/·125初/ps自cl_loss,5100,csc/bg无类名,头名,s机,M=16/' \
                        'checkpoints/final.pth'

cfg.dt.val.ini.split = 'val'

cfg.cls.fg_num = IL(lambda c: c.dt.val.dt.class_num - 1)
cfg.cls.mul_factor = Param2Tune([0.25, 0.5, 0.75, 1])
cfg.cls.thresh = Param2Tune([.01, .04, .08, .12, .16, .20])
