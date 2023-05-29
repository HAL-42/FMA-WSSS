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

# * 配置CAM路径。
cfg.cam.dir = 'experiment/clip_cam/离线伪真,CI/l1/·125初/ps自cl_loss,5100,s性,M=8/infer/final,val/cam'

# * 配置替补种子点路径。
cfg.seed.dir = 'experiment/clip_cam/离线伪真,CI/l1/·125初/' \
               'ps自cl_loss,5100,s性,M=8/infer/final,val/aff2次,at_cam,att1次,·5掩阈,ce_npp/seed/best/mask'

# * 配置SAM标注路径。
cfg.sam_anns.pattern_key = Param2Tune(['l2_nmsf_s1_rsw3',
                                       'ssa_default',
                                       'l2_only_s1',
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
@cfg.sam_anns.set_IL()  # noqa
def dir(c: Config):  # noqa
    return f'experiment/sam_auto_seg/vh,val/pattern_key={c.sam_anns.pattern_key}/anns'

# * 配置种子生成参数。
cfg.seed.ini.norm_first = True
cfg.seed.ini.gather_method = 'mean'
cfg.seed.ini.bg_method = Param2Tune([{'method': 'pow', 'pow': .8},
                                     {'method': 'pow', 'pow': .9},
                                     {'method': 'pow', 'pow': 1},
                                     {'method': 'pow', 'pow': 2},
                                     {'method': 'pow', 'pow': 3},
                                     {'method': 'pow', 'pow': 4}],
                                    optional_value_names=['p.8', 'p.9', 'p1', 'p2', 'p3', 'p4'])
cfg.seed.ini.priority = Param2Tune([('conf_bigger',),
                                    ('level_bigger', 'conf_bigger'),
                                    ('level_smaller', 'conf_bigger')])
