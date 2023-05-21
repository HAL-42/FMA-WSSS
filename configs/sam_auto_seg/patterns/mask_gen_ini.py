#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/18 13:39
@File    : mask_gen_ini.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config()

# * 官方参数。
cfg.official_default.points_per_side = 32

cfg.official_default.pred_iou_thresh = 0.88
cfg.official_default.stability_score_thresh = 0.95
cfg.official_default.stability_score_offset = 1.0
cfg.official_default.box_nms_thresh = 0.7

cfg.official_default.crop_n_layers = 0
cfg.official_default.crop_nms_thresh = 0.7
cfg.official_default.crop_overlap_ratio = 512 / 1500
cfg.official_default.crop_n_points_downscale_factor = 1

cfg.official_default.min_mask_region_area = 0

cfg.official_default.score_thresh_offset = (0.0, 0.0, 0.0)
cfg.official_default.score_nms_offset = (0.0, 0.0, 0.0)
cfg.official_default.stability_score_bias = (0.0, 0.0, 0.0)
cfg.official_default.rock_sand_water_thresh = None

# * 官方重型。
cfg.official_heavy.points_per_side = 32

cfg.official_heavy.pred_iou_thresh = 0.86
cfg.official_heavy.stability_score_thresh = 0.92
cfg.official_heavy.stability_score_offset = 1.0
cfg.official_heavy.box_nms_thresh = 0.7

cfg.official_heavy.crop_n_layers = 1
cfg.official_heavy.crop_nms_thresh = 0.7
cfg.official_heavy.crop_overlap_ratio = 512 / 1500
cfg.official_heavy.crop_n_points_downscale_factor = 2

cfg.official_heavy.min_mask_region_area = 100

cfg.official_heavy.score_thresh_offset = (0.0, 0.0, 0.0)
cfg.official_heavy.score_nms_offset = (0.0, 0.0, 0.0)
cfg.official_heavy.stability_score_bias = (0.0, 0.0, 0.0)
cfg.official_heavy.rock_sand_water_thresh = None

# * SSA Light
cfg.ssa_light.points_per_side = 16

cfg.ssa_light.pred_iou_thresh = 0.86
cfg.ssa_light.stability_score_thresh = 0.92
cfg.ssa_light.stability_score_offset = 1.0
cfg.ssa_light.box_nms_thresh = 0.7

cfg.ssa_light.crop_n_layers = 0
cfg.ssa_light.crop_nms_thresh = 0.7
cfg.ssa_light.crop_overlap_ratio = 512 / 1500
cfg.ssa_light.crop_n_points_downscale_factor = 2

cfg.ssa_light.min_mask_region_area = 100

cfg.ssa_light.score_thresh_offset = (0.0, 0.0, 0.0)
cfg.ssa_light.score_nms_offset = (0.0, 0.0, 0.0)
cfg.ssa_light.stability_score_bias = (0.0, 0.0, 0.0)
cfg.ssa_light.rock_sand_water_thresh = None

# * SSA Light 8点
cfg.ssa_light_8p.points_per_side = 8

cfg.ssa_light_8p.pred_iou_thresh = 0.86
cfg.ssa_light_8p.stability_score_thresh = 0.92
cfg.ssa_light_8p.stability_score_offset = 1.0
cfg.ssa_light_8p.box_nms_thresh = 0.7

cfg.ssa_light_8p.crop_n_layers = 0
cfg.ssa_light_8p.crop_nms_thresh = 0.7
cfg.ssa_light_8p.crop_overlap_ratio = 512 / 1500
cfg.ssa_light_8p.crop_n_points_downscale_factor = 2

cfg.ssa_light_8p.min_mask_region_area = 100

cfg.ssa_light_8p.score_thresh_offset = (0.0, 0.0, 0.0)
cfg.ssa_light_8p.score_nms_offset = (0.0, 0.0, 0.0)
cfg.ssa_light_8p.stability_score_bias = (0.0, 0.0, 0.0)
cfg.ssa_light_8p.rock_sand_water_thresh = None

# * SSA Default
cfg.ssa_default.points_per_side = 32

cfg.ssa_default.pred_iou_thresh = 0.86
cfg.ssa_default.stability_score_thresh = 0.92
cfg.ssa_default.stability_score_offset = 1.0
cfg.ssa_default.box_nms_thresh = 0.7

cfg.ssa_default.crop_n_layers = 0
cfg.ssa_default.crop_nms_thresh = 0.7
cfg.ssa_default.crop_overlap_ratio = 512 / 1500
cfg.ssa_default.crop_n_points_downscale_factor = 2

cfg.ssa_default.min_mask_region_area = 100

cfg.ssa_default.score_thresh_offset = (0.0, 0.0, 0.0)
cfg.ssa_default.score_nms_offset = (0.0, 0.0, 0.0)
cfg.ssa_default.stability_score_bias = (0.0, 0.0, 0.0)
cfg.ssa_default.rock_sand_water_thresh = None

# * SSA Heavy
cfg.ssa_heavy.points_per_side = 64

cfg.ssa_heavy.pred_iou_thresh = 0.86
cfg.ssa_heavy.stability_score_thresh = 0.92
cfg.ssa_heavy.stability_score_offset = 1.0
cfg.ssa_heavy.box_nms_thresh = 0.7

cfg.ssa_heavy.crop_n_layers = 1
cfg.ssa_heavy.crop_nms_thresh = 0.7
cfg.ssa_heavy.crop_overlap_ratio = 512 / 1500
cfg.ssa_heavy.crop_n_points_downscale_factor = 2

cfg.ssa_heavy.min_mask_region_area = 100

cfg.ssa_heavy.score_thresh_offset = (0.0, 0.0, 0.0)
cfg.ssa_heavy.score_nms_offset = (0.0, 0.0, 0.0)
cfg.ssa_heavy.stability_score_bias = (0.0, 0.0, 0.0)
cfg.ssa_heavy.rock_sand_water_thresh = None

# * SAM Auto阈值滤除分分、分。
cfg.l2_only.points_per_side = 32

cfg.l2_only.pred_iou_thresh = 0.86
cfg.l2_only.stability_score_thresh = 0.92
cfg.l2_only.stability_score_offset = 1.0
cfg.l2_only.box_nms_thresh = 0.7

cfg.l2_only.crop_n_layers = 1
cfg.l2_only.crop_nms_thresh = 0.7
cfg.l2_only.crop_overlap_ratio = 512 / 1500
cfg.l2_only.crop_n_points_downscale_factor = 2

cfg.l2_only.min_mask_region_area = 100

cfg.l2_only.score_thresh_offset = (-1.0, -1.0, 0.0)
cfg.l2_only.score_nms_offset = (0.0, 0.0, 0.0)
cfg.l2_only.stability_score_bias = (0.0, 0.0, 0.0)
cfg.l2_only.rock_sand_water_thresh = None

# * SAM Auto阈值滤除分分、分，并提高总级的阈值得分偏移。
cfg.l2_only_s1.points_per_side = 32

cfg.l2_only_s1.pred_iou_thresh = 0.86
cfg.l2_only_s1.stability_score_thresh = 0.92
cfg.l2_only_s1.stability_score_offset = 1.0
cfg.l2_only_s1.box_nms_thresh = 0.7

cfg.l2_only_s1.crop_n_layers = 1
cfg.l2_only_s1.crop_nms_thresh = 0.7
cfg.l2_only_s1.crop_overlap_ratio = 512 / 1500
cfg.l2_only_s1.crop_n_points_downscale_factor = 2

cfg.l2_only_s1.min_mask_region_area = 100

cfg.l2_only_s1.score_thresh_offset = (-1.0, -1.0, 0.1)
cfg.l2_only_s1.score_nms_offset = (0.0, 0.0, 0.0)
cfg.l2_only_s1.stability_score_bias = (0.0, 0.0, 0.0)
cfg.l2_only_s1.rock_sand_water_thresh = None

# * SAM Auto阈值滤除分分、分，并提高总级的阈值得分偏移，降低总的稳定度阈值。
cfg.l2_only_s1_t4.points_per_side = 32

cfg.l2_only_s1_t4.pred_iou_thresh = 0.86
cfg.l2_only_s1_t4.stability_score_thresh = 0.92
cfg.l2_only_s1_t4.stability_score_offset = 1.0
cfg.l2_only_s1_t4.box_nms_thresh = 0.7

cfg.l2_only_s1_t4.crop_n_layers = 1
cfg.l2_only_s1_t4.crop_nms_thresh = 0.7
cfg.l2_only_s1_t4.crop_overlap_ratio = 512 / 1500
cfg.l2_only_s1_t4.crop_n_points_downscale_factor = 2

cfg.l2_only_s1_t4.min_mask_region_area = 100

cfg.l2_only_s1_t4.score_thresh_offset = (-1.0, -1.0, 0.1)
cfg.l2_only_s1_t4.score_nms_offset = (0.0, 0.0, 0.0)
cfg.l2_only_s1_t4.stability_score_bias = (0.0, 0.0, 0.04)
cfg.l2_only_s1_t4.rock_sand_water_thresh = None

# * SAM Auto总NMS优先。
cfg.l2_nmsf.points_per_side = 32

cfg.l2_nmsf.pred_iou_thresh = 0.86
cfg.l2_nmsf.stability_score_thresh = 0.92
cfg.l2_nmsf.stability_score_offset = 1.0
cfg.l2_nmsf.box_nms_thresh = 0.7

cfg.l2_nmsf.crop_n_layers = 1
cfg.l2_nmsf.crop_nms_thresh = 0.7
cfg.l2_nmsf.crop_overlap_ratio = 512 / 1500
cfg.l2_nmsf.crop_n_points_downscale_factor = 2

cfg.l2_nmsf.min_mask_region_area = 100

cfg.l2_nmsf.score_thresh_offset = (0.0, 0.0, 0.0)
cfg.l2_nmsf.score_nms_offset = (0.0, 0.0, 1.0)
cfg.l2_nmsf.stability_score_bias = (0.0, 0.0, 0.0)
cfg.l2_nmsf.rock_sand_water_thresh = None

# * SAM Auto总NMS优先，提高总级的阈值得分偏移。
cfg.l2_nmsf_s1.points_per_side = 32

cfg.l2_nmsf_s1.pred_iou_thresh = 0.86
cfg.l2_nmsf_s1.stability_score_thresh = 0.92
cfg.l2_nmsf_s1.stability_score_offset = 1.0
cfg.l2_nmsf_s1.box_nms_thresh = 0.7

cfg.l2_nmsf_s1.crop_n_layers = 1
cfg.l2_nmsf_s1.crop_nms_thresh = 0.7
cfg.l2_nmsf_s1.crop_overlap_ratio = 512 / 1500
cfg.l2_nmsf_s1.crop_n_points_downscale_factor = 2

cfg.l2_nmsf_s1.min_mask_region_area = 100

cfg.l2_nmsf_s1.score_thresh_offset = (0.0, 0.0, 0.1)
cfg.l2_nmsf_s1.score_nms_offset = (0.0, 0.0, 1.0)
cfg.l2_nmsf_s1.stability_score_bias = (0.0, 0.0, 0.0)
cfg.l2_nmsf_s1.rock_sand_water_thresh = None

# * SAM Auto总NMS优先，提高总级的阈值得分偏移，crop0次。
cfg.l2_nmsf_s1_c0.points_per_side = 32

cfg.l2_nmsf_s1_c0.pred_iou_thresh = 0.86
cfg.l2_nmsf_s1_c0.stability_score_thresh = 0.92
cfg.l2_nmsf_s1_c0.stability_score_offset = 1.0
cfg.l2_nmsf_s1_c0.box_nms_thresh = 0.7

cfg.l2_nmsf_s1_c0.crop_n_layers = 0
cfg.l2_nmsf_s1_c0.crop_nms_thresh = 0.7
cfg.l2_nmsf_s1_c0.crop_overlap_ratio = 512 / 1500
cfg.l2_nmsf_s1_c0.crop_n_points_downscale_factor = 2

cfg.l2_nmsf_s1_c0.min_mask_region_area = 100

cfg.l2_nmsf_s1_c0.score_thresh_offset = (0.0, 0.0, 0.1)
cfg.l2_nmsf_s1_c0.score_nms_offset = (0.0, 0.0, 1.0)
cfg.l2_nmsf_s1_c0.stability_score_bias = (0.0, 0.0, 0.0)
cfg.l2_nmsf_s1_c0.rock_sand_water_thresh = None

# * SAM Auto总NMS优先，提高总级的阈值得分偏移，结合石沙水过滤。
cfg.l2_nmsf_s1_rsw3.points_per_side = 32

cfg.l2_nmsf_s1_rsw3.pred_iou_thresh = 0.86
cfg.l2_nmsf_s1_rsw3.stability_score_thresh = 0.92
cfg.l2_nmsf_s1_rsw3.stability_score_offset = 1.0
cfg.l2_nmsf_s1_rsw3.box_nms_thresh = 0.7

cfg.l2_nmsf_s1_rsw3.crop_n_layers = 1
cfg.l2_nmsf_s1_rsw3.crop_nms_thresh = 0.7
cfg.l2_nmsf_s1_rsw3.crop_overlap_ratio = 512 / 1500
cfg.l2_nmsf_s1_rsw3.crop_n_points_downscale_factor = 2

cfg.l2_nmsf_s1_rsw3.min_mask_region_area = 100

cfg.l2_nmsf_s1_rsw3.score_thresh_offset = (0.0, 0.0, 0.1)
cfg.l2_nmsf_s1_rsw3.score_nms_offset = (0.0, 0.0, 1.0)
cfg.l2_nmsf_s1_rsw3.stability_score_bias = (0.0, 0.0, 0.0)
cfg.l2_nmsf_s1_rsw3.rock_sand_water_thresh = 0.3
