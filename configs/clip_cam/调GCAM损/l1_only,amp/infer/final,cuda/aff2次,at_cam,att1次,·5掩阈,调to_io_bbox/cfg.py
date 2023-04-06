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
                        cfgs_update_at_parser=('configs/aff_voc/base.py',))

cfg.rslt_dir = ...

# * 覆盖原配置，使其适合调参（不改变算法）。
cfg.aff.ori_cam_dir = 'experiment/clip_cam/调GCAM损/l1_only,amp/infer/final,cuda/cam'

# cfg.solver.viz_cam = False  # noqa
# cfg.solver.viz_score = False

# * 修改算法参数。
cfg.aff.ini.att2aff_cfg.method.n_iter = 1
cfg.aff.ini.aff_mask_cfg.method.thresh = .5

cfg.aff.ini.aff_cfg.n_iters = 2
cfg.aff.ini.aff_at = 'cam'

# * 调整mask的关注范围。
cfg.aff.ini.aff_mask_cfg.method.to_in_bbox = Param2Tune(['in_bbox', 'all'])
cfg.aff.ini.aff_mask_cfg.method.to_out_bbox = Param2Tune(['in_bbox', 'out_bbox', 'all'])
