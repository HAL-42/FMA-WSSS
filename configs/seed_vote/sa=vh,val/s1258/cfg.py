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
from alchemy_cat.py_tools import Cfg2Tune, Param2Tune, Config

cfg = config = Cfg2Tune('configs/patterns/seg_metrics/cls_m_IoU,pra.py',
                        cfgs_update_at_parser=('configs/seed_vote/base.py',))

cfg.rslt_dir = ...

# * 在val上推理。
cfg.dt.ini.split = 'val'

# * 选取pattern key。
cfg.sam_anns.pattern_key = Param2Tune(['ssa_default',
                                       'l2_only_s1',
                                       'l2_nmsf_s1_rsw3',
                                       'l2_nmsf_s1',
                                       'l2_only_s1_t4',
                                       'l2_nmsf_s1_c0',
                                       'l2_nmsf',
                                       'l2_only',
                                       'ssa_light',
                                       'ssa_light_8p',
                                       'official_heavy',
                                       'ssa_heavy',
                                       'official_default',
                                       ])

# * 指定种子和标注位置。
cfg.seed.dir = 'experiment/clip_cam/离线伪真,CI/l1/·125初/' \
               'ps自cl_loss,5100,s性,M=8/infer/final,val/aff2次,at_cam,att1次,·5掩阈,ce_npp/seed/best/mask'
@cfg.sam_anns.set_IL()  # noqa
def dir(c: Config):  # noqa
    return f'experiment/sam_auto_seg/vh,val/pattern_key={c.sam_anns.pattern_key}/anns'

# * 选择模型参数。
cfg.voter.ini.sam_seg_occupied_by_fg_thresh = Param2Tune([0.5, 0.4, 0.6])  # noqa
cfg.voter.ini.fg_occupied_by_sam_seg_thresh = Param2Tune([0.85, 0.75, 0.95])
cfg.voter.ini.use_seed_when_no_sam = Param2Tune([True, False])
