#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/25 16:05
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Cfg2Tune, Param2Tune, Config

cfg = config = Cfg2Tune('configs/patterns/seg_metrics/cls_m_IoU,pra.py',
                        cfgs_update_at_parser=('configs/anns_seed/base.py',))

cfg.rslt_dir = ...

cfg.dt.ini.split = 'train'

# * 配置SAM标注路径。
cfg.sam_anns.pattern_key = Param2Tune(['l2_nmsf_s1_rsw3', 'ssa_heavy'])

@cfg.sam_anns.set_IL()  # noqa
def dir(c: Config):  # noqa
    return f'experiment/sam_auto_seg/vh,t/pattern_key={c.sam_anns.pattern_key}/anns'

# * 配置CAM路径。
cfg.cam.dir = 'experiment/clip_cam/离线伪真,CI/l1/·125初/ps自真152a,csc/bg无类名,头名,s机,M=16/infer/final,train/' \
              'aff2次,at_cam,att1次,·5掩阈/cam_affed'

# * 修改得分算法参数。
cfg.seed.norm_firsts = [True]
cfg.seed.bg_methods = [{'method': 'pow', 'pow': .5}]

# * 修改可视化配置。
cfg.viz.step = 1
