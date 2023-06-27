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
import os.path as osp

from alchemy_cat.py_tools import Cfg2Tune, Param2Tune

cfg = config = Cfg2Tune('configs/patterns/seg_metrics/cls_m_IoU,pra.py',
                        cfgs_update_at_parser=('configs/aff_voc/coco/base.py',))

cfg.rslt_dir = ...

# * 调参特有配置。
@cfg.aff.set_IL()  # noqa
def ori_cam_dir(c):
    return osp.join(c.rslt_dir, '..', '..', 'cam')  # 因为是调参配置，在上两级目录中寻找cam。

cfg.solver.save_cam = False
cfg.solver.viz_cam = False

# * 调整参数。
cfg.aff.ini.att2aff_cfg.method.n_iter = 1  # noqa
cfg.aff.ini.aff_mask_cfg.method.thresh = Param2Tune([.5, .6, .7, .8])
cfg.aff.ini.aff_cfg.n_iters = 1
cfg.aff.ini.aff_at = 'cam'

# * 设定eval方法。
cfg.eval.seed.bg_methods = [{'method': 'pow', 'pow': p} for p in range(2, 5)]
