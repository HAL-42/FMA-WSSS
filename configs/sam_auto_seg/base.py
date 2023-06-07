#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/17 22:39
@File    : base.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config, IL

from libs.data import VOCAug2
from libs.sam import SamAuto

cfg = config = Config()

cfg.rslt_dir = ...
cfg.rand_seed = 0  # 与随机参考使用相同的随机种子。如此相比基线多出随机部分，参考不同基线时，有不同的随机性。

# * 配置数据集。
cfg.dt.ini.root = 'datasets'
cfg.dt.ini.split = 'train_aug'
cfg.dt.ini.cls_labels_type = 'seg_cls_labels'
cfg.dt.ini.ps_mask_dir = None
cfg.dt.ini.rgb_img = True
cfg.dt.cls = VOCAug2

# * 配置sam基础模型。
cfg.sam.model_type = 'vit_h'
cfg.sam.checkpoint = IL(lambda c:
                        {'vit_h': 'pretrains/SAM/sam_vit_h_4b8939.pth',
                         'vit_l': 'pretrains/SAM/sam_vit_l_0b3195.pth',
                         'vit_b': 'pretrains/SAM/sam_vit_b_01ec64.pth'}[c.sam.model_type]
                        )

# * 配置基于SAM的掩码生成器。
# cfg.mask_gen.ini.points_per_side = 32
# cfg.mask_gen.ini.points_per_batch = 256
# cfg.mask_gen.ini.pred_iou_thresh = 0.86
# cfg.mask_gen.ini.stability_score_thresh = 0.92
# cfg.mask_gen.ini.stability_score_offset = 1.0
# cfg.mask_gen.ini.box_nms_thresh = 0.7
# cfg.mask_gen.ini.crop_n_layers = 1
# cfg.mask_gen.ini.crop_nms_thresh = 0.7
# cfg.mask_gen.ini.crop_overlap_ratio = 512 / 1500
# cfg.mask_gen.ini.crop_n_points_downscale_factor = 2
# cfg.mask_gen.ini.min_mask_region_area = 100
# cfg.mask_gen.ini.output_mode = "binary_mask"
# cfg.mask_gen.ini.score_thresh_offset = (0.0, 0.0, 0.1)
# cfg.mask_gen.ini.score_nms_offset = (0.0, 0.0, 1.0)
# cfg.mask_gen.ini.stability_score_bias = (0.0, 0.0, 0.0)
# cfg.mask_gen.ini.rock_sand_water_thresh = 0.3
# cfg.mask_gen.ini.rock_sand_water_chunk_size = 50
cfg.mask_gen.ini = ...

cfg.mask_gen.cls = SamAuto

# * 保存与可视化。
cfg.viz.enable = True
cfg.viz.level_combs = [(0, 1, 2), (0, 1), (2,)]
cfg.viz.alpha = 0.5
cfg.viz.step = 100
