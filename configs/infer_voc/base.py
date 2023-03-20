#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/10 21:27
@File    : cfg.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.alg import find_nearest_even_size
from alchemy_cat.py_tools import Config, IL

from libs import coop
from libs import io
from libs.data import VOCAug2, VOC2Auger
from libs.seeding.seed_argmax import seed_argmax

cfg = config = Config('configs/patterns/voc_names/clip_es.py')

cfg.rslt_dir = ...
cfg.rand_seed = 0  # 与随机参考使用相同的随机种子。如此相比基线多出随机部分，参考不同基线时，有不同的随机性。

# * 设定数据集。
dt = cfg.dt
# ** 设定验证集。
dt.val.ini.cls_labels_type = 'seg_cls_labels'
dt.val.dt = IL(lambda c:
               VOCAug2(root='datasets', split='train_aug', **c.dt.val.ini),
               priority=-1)

# * 设定训练和测试数据增强器。
auger = cfg.auger

auger.val.ini.is_color_jitter = False
scale_crop_method = auger.val.ini.scale_crop_method
scale_crop_method.method = 'scale_align'
scale_crop_method.aligner = find_nearest_even_size
scale_crop_method.scale_factors = [1.]
auger.val.ini.is_rand_mirror = False
auger.val.ini.mean = (0.48145466, 0.4578275, 0.40821073)
auger.val.ini.std = (0.26862954, 0.26130258, 0.27577711)
auger.val.ini.lb_scale_factor = None
auger.val.ini.ol_cls_lb = False
auger.val.cls = VOC2Auger

# * 设定数据管理器。
cfg.loader.val.batch_size = 1
cfg.loader.val.num_workers = 12

# * 设定网络的输入处理。
cfg.io.update_inp = io.voc_inp_to_gcam_clip  # 增加inp.fg_cls_lb。
cfg.io.update_out = io.gcam_clip_out_to_cls_loss  # 增加out.fg_logits。

# * 设定网络。
model = cfg.model

model.ini.clip_name = 'ViT-B/16'
model.ini.fp32 = True
model.ini.classnames = IL(lambda c: c.model.fg_names + c.model.bg_names)
model.ini.ctx_cfg.n_ctx = 16
model.ini.ctx_cfg.ctx_init = ''
model.ini.ctx_cfg.csc = False
model.ini.ctx_cfg.cls_token_pos = 'end'
model.ini.adaptive_pos_emb = True
model.ini.sm_fg_exist = True
model.cls = coop.grad_cam_clip

model.val_cal = lambda m, inp: m(inp.img, inp.fg_cls_lb)
model.resume_file = ''

# * 设定保存的内容。
cfg.solver.save_cam = True
cfg.solver.viz_cam = True
cfg.solver.viz_score.resize_first = IL(lambda c: c.eval.seed.resize_first)
cfg.solver.viz_step = 100

# * 设定eval方法。
cfg.eval.enabled = True
cfg.eval.seed.cal = seed_argmax
cfg.eval.seed.bg_methods = [{'method': 'pow', 'pow': p} for p in range(2, 9)]
cfg.eval.seed.resize_first = True  # 先阈值+归一化，还是先resize。
