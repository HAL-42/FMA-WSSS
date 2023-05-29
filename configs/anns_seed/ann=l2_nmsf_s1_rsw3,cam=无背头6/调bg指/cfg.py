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
from alchemy_cat.py_tools import Cfg2Tune, Param2Tune

cfg = config = Cfg2Tune('configs/patterns/seg_metrics/cls_m_IoU,pra.py',
                        cfgs_update_at_parser=('configs/anns_seed/base.py',))

cfg.rslt_dir = ...

# * 配置CAM路径。
cfg.cam.dir = 'experiment/clip_cam/离线伪真,CI/l1/·125初/ps自cl_loss,5100,csc/bg无类名,头名,s机,M=16/infer/final,val/cam'

# * 配置替补种子点路径。
cfg.seed.dir = 'experiment/clip_cam/离线伪真,CI/l1/·125初/ps自cl_loss,5100,csc/bg无类名,头名,s机,M=16/infer/final,val/' \
               'aff2次,at_cam,att1次,·5掩阈,ce_npp/seed/best/mask'

# * 配置SAM标注路径。
cfg.sam_anns.dir = 'experiment/sam_auto_seg/vh,val/pattern_key=l2_nmsf_s1_rsw3/anns'

# * 配置种子生成参数。
cfg.seed.ini.norm_first = Param2Tune([False])
cfg.seed.ini.gather_method = Param2Tune(['mean'])
cfg.seed.ini.bg_method = Param2Tune([{'method': 'pow', 'pow': .8},
                                     {'method': 'pow', 'pow': .9},
                                     {'method': 'pow', 'pow': 1},
                                     {'method': 'pow', 'pow': 2}],
                                    optional_value_names=['p.8', 'p.9', 'p1', 'p2'])
cfg.seed.ini.priority = Param2Tune([('level_bigger', 'conf_bigger')])
