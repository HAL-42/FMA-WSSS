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
from alchemy_cat.py_tools import Cfg2Tune, Param2Tune

cfg = config = Cfg2Tune('configs/patterns/seg_metrics/cls_m_IoU,pra.py',
                        cfgs_update_at_parser=('configs/infer_voc/align/base.py',))

cfg.rslt_dir = ...

cfg.solver.save_cam = False  # 节省空间，不保存cam。

cfg.model.resume_file = Param2Tune(['experiment/clip_cam/调cls/cl_loss/checkpoints/iter-3400.pth',
                                    'experiment/clip_cam/调cls/cl_loss/checkpoints/iter-5100.pth',
                                    'experiment/clip_cam/调cls/cl_loss/checkpoints/iter-6800.pth',
                                    'experiment/clip_cam/调cls/cl_loss/checkpoints/iter-10200.pth',
                                    'experiment/clip_cam/调cls/cl_loss/checkpoints/iter-13600.pth',
                                    'experiment/clip_cam/调cls/cl_loss/checkpoints/iter-17000.pth',
                                    'experiment/clip_cam/调cls/cl_loss/checkpoints/final.pth'],
                                   optional_value_names=['3400',
                                                         '5100',
                                                         '6800',
                                                         '10200',
                                                         '13600',
                                                         '17000',
                                                         'final'])
