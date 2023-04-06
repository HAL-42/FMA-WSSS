#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/29 16:14
@File    : cfg.py
@Software: PyCharm
@Desc    : 在 configs/clip_cam/调GCAM损/l1_only,amp/infer/final,cuda/调aff_at,aff次数 最优结果上，调整att2aff的迭代次数，
           以及mask阈值。
"""
import os.path as osp

from alchemy_cat.py_tools import Cfg2Tune, Param2Tune

cfg = config = Cfg2Tune('configs/patterns/seg_metrics/cls_m_IoU,pra.py',
                        cfgs_update_at_parser=('configs/aff_voc/base.py',))

cfg.rslt_dir = ...

# * 覆盖原配置，使其适合调参（不改变算法）。
@cfg.aff.set_IL()  # noqa
def ori_cam_dir(c):
    return osp.join(c.rslt_dir, '..', '..', 'cam')  # 因为是调参配置，在上两级目录中寻找cam。

cfg.solver.viz_cam = False  # noqa
cfg.solver.viz_score = False

# * 修改算法参数。
cfg.aff.ini.att2aff_cfg.method.n_iter = Param2Tune([1, 2, 3, 4])
cfg.aff.ini.aff_mask_cfg.method.thresh = Param2Tune([.3, .4, .5, .6, .7])

cfg.aff.ini.aff_cfg.n_iters = 2
cfg.aff.ini.aff_at = 'score'
