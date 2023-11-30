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
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/patterns/seg_metrics/cls_m_IoU,pra.py',
                      cfgs_update_at_parser=('configs/seed_vote/base.py',))

cfg.rslt_dir = ...

# * 在val上推理。
cfg.dt.ini.split = 'train_aug'

# * 选取pattern key。
cfg.sam_anns.pattern_key = 'l2_nmsf_s1_rsw3'

# * 指定种子和标注位置。
cfg.seed.dir = 'datasets/VOC2012/SegmentationClassAug'
@cfg.sam_anns.set_IL()  # noqa
def dir(c: Config):  # noqa
    return f'experiment/sam_auto_seg/vh,ta/pattern_key={c.sam_anns.pattern_key}/anns'

# * 选择模型参数。
cfg.voter.ini.sam_seg_occupied_by_fg_thresh = 0.5
cfg.voter.ini.fg_occupied_by_sam_seg_thresh = 0.85
cfg.voter.ini.use_seed_when_no_sam = True
