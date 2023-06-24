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
cfg.aff.ini.att2aff_cfg.method.n_iter = Param2Tune([1, 2, 3])  # noqa
cfg.aff.ini.aff_mask_cfg.method.thresh = .6
cfg.aff.ini.aff_cfg.n_iters = 2
cfg.aff.ini.aff_at = Param2Tune(['cam', 'score'])

# * 设定eval方法。
cfg.eval.seed.bg_methods = [{'method': 'pow', 'pow': p} for p in range(1, 3)]
