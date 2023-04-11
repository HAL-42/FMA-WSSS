#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/10 12:16
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.contrib.schedulers import WarmPolynomialLR
from alchemy_cat.py_tools import Cfg2Tune, Param2Tune, Config, IL

from utils.lr_scheduler import CosineAnnealingLR

cfg = config = Cfg2Tune('configs/patterns/seg_metrics/cls_m_IoU,pra.py',
                        cfgs_update_at_parser=('configs/clip_cam/cl_loss,调基参/base.py',))

cfg.rslt_dir = ...

# * 最小学习率是初始学习率的0.8。
def main_iter(c):  # noqa
    return c.solver.max_iter - c.sched.warm.warm_iters

cos_sched = Config()  # noqa
cos_sched._whole = True
cos_sched._param_val_name = 'CosineAnnealingCfg'
cos_sched.ini.T_max = IL(main_iter)
cos_sched.ini.eta_min = 0.001 * 0.8
cos_sched.cls = CosineAnnealingLR

poly_sched = Config()
poly_sched._whole = True
poly_sched._param_val_name = 'WarmPolynomialCfg'
poly_sched.ini.max_iter = IL(main_iter)
poly_sched.ini.end_lr_factors = 0.8
poly_sched.cls = WarmPolynomialLR

poly7_sched = poly_sched.branch_copy()
poly7_sched._param_val_name = 'WarmPolynomialCfg_p7'
poly7_sched.ini.power = 0.7

poly5_sched = poly_sched.branch_copy()
poly5_sched._param_val_name = 'WarmPolynomialCfg_p5'
poly5_sched.ini.power = 0.5

cfg.sched.main = Param2Tune([cos_sched, poly_sched, poly7_sched, poly5_sched])

# * 一律预热500迭代（0.75）轮，训练5100迭代。
cfg.solver.max_iter = 5100
