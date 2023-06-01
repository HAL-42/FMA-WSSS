#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/8 23:01
@File    : base.py
@Software: PyCharm
@Desc    : 
"""
import re

from alchemy_cat.py_tools import Config, IL
from torch import nn

from libs import coop
from libs import io
from libs.data import VOCAug2, VOC2Auger
from libs.loss import cam_lb
from utils.lr_scheduler import CosineAnnealingLR, LinearLR

cfg = config = Config('configs/patterns/voc_names/clip_es.py')

cfg.rslt_dir = ...
cfg.rand_seed = 0  # 与随机参考使用相同的随机种子。如此相比基线多出随机部分，参考不同基线时，有不同的随机性。

# * 设定随机参考。
cfg.rand_ref.ref_dir = None
cfg.rand_ref.rand_copy = {}

# * 设定数据集。
dt = cfg.dt
# ** 设定训练集。
dt.train.ini.cls_labels_type = 'seg_cls_labels'
dt.train.ini.ps_mask_dir = None
dt.train.dt = IL(lambda c:
                 VOCAug2(root='datasets', split='train_aug', **c.dt.train.ini),
                 priority=-10)

dt.train.epoch_len = IL(lambda c:
                        len(c.dt.train.dt) // c.loader.train.batch_size,
                        priority=0)
# ** 设定丁真集。
# dt.few_shot.ini.shot_num = 16
# dt.few_shot.ini.seed = IL(lambda c: c.rand_seed)
# dt.few_shot.ini.except_bg = False
# dt.few_shot.dt = IL(lambda c:
#                     FewShotDt(c.train.dt, **c.few_shot.ini),
#                     priority=0
#                     )
#
# dt.few_shot.epoch_len = IL(lambda c:
#                            len(c.dt.few_shot.dt) // c.loader.train.batch_size,
#                            )
dt.few_shot.dt = IL(lambda c: c.dt.train.dt)  # 先用完整数据集训练。
dt.few_shot.epoch_len = IL(lambda c: c.dt.train.epoch_len)

# * 设定训练和测试数据增强器。
auger = cfg.auger

auger.train.ini.is_color_jitter = True
scale_crop_method = auger.train.ini.scale_crop_method
scale_crop_method.method = 'rand_range'
scale_crop_method.low_size = 288
scale_crop_method.high_size = 480
scale_crop_method.short_thresh = 224  # 希望短边从224~(224 * 1.5)，长短比≈1.4，算出长边范围。
scale_crop_method.crop_size = 224
auger.train.ini.is_rand_mirror = True
auger.train.ini.mean = (0.48145466, 0.4578275, 0.40821073)
auger.train.ini.std = (0.26862954, 0.26130258, 0.27577711)
auger.train.ini.lb_scale_factor = None
auger.train.ini.ol_cls_lb = True
auger.train.cls = VOC2Auger

# * 设定数据管理器。
cfg.loader.train.batch_size = 16
cfg.loader.train.sub_iter_num = 1
cfg.loader.num_workers = 12

# * 设定网络的输入处理。
cfg.io.update_inp = io.voc_inp_to_gcam_clip  # 增加inp.fg_cls_lb。
cfg.io.update_out = io.gcam_clip_out_to_cls_loss  # 增加out.fg_logits。

# * 设定网络。
model = cfg.model

model.patch_size = IL(lambda c: int(re.search(r'/(\d+)', c.model.ini.clip_name).group(1)))

model.ini.clip_name = 'ViT-B/16'
model.ini.fp32 = False
model.ini.classnames = IL(lambda c: c.model.fg_names + c.model.bg_names)  # ! 只用于初始化prompt leaner，并非真实类名。
model.ini.ctx_cfg.n_ctx = 16
model.ini.ctx_cfg.ctx_init = ''
model.ini.ctx_cfg.csc = False
model.ini.ctx_cfg.cls_token_pos = 'end'
model.ini.ctx_cfg.ctx_std = 0.0125
model.ini.adaptive_pos_emb = True
model.ini.sm_fg_exist = True
model.cls = coop.grad_cam_clip

def model_cal(m, inp):  # noqa
    return m(inp.img, inp.fg_cls_lb, pad_info=inp.pad_info if inp.pad_info else None)
model.cal = model_cal  # noqa

model.initialize_seed = IL(lambda c: c.rand_seed)

# * 设定优化器。
opt = cfg.opt
opt.get_pg.ini.lr = IL(lambda c: c.opt.base_lr)
opt.get_pg.ini.weight_decay = 5e-4
opt.base_lr = 0.001
opt.momentum = 0.9

# * 设定Scheduler。
sched = cfg.sched

sched.warm.warm_iters = 500  # 0.75轮。
sched.warm.ini.start_factor = IL(lambda c:
                                 1e-5 / c.opt.base_lr,
                                 priority=0)
sched.warm.ini.end_factor = IL(lambda c: c.sched.warm.ini.start_factor)
sched.warm.ini.total_iters = IL(lambda c: c.sched.warm.warm_iters)
sched.warm.cls = LinearLR

sched.main.ini.T_max = IL(lambda c: c.solver.max_iter - c.sched.warm.warm_iters)
sched.main.ini.eta_min = 0.
sched.main.cls = CosineAnnealingLR

# * 设定loss函数。
loss_items = cfg.loss.loss_items

loss_items.multi_cls.cri = nn.MultiLabelSoftMarginLoss(reduction='mean')
loss_items.multi_cls.cal = lambda cri, inp, out: cri(out.fg_logits, inp.fg_cls_lb)
loss_items.multi_cls.weights = 0.1

loss_items.cam_lb.ini.loss_type = 'l1'
loss_items.cam_lb.ini.reduce = 'all'
loss_items.cam_lb.ini.detach_max = True
loss_items.cam_lb.ini.bg_thresh = 0.
loss_items.cam_lb.cri = IL(lambda c: cam_lb.CAMIntensityLoss(**c.loss.loss_items.cam_lb.ini))
loss_items.cam_lb.cal = lambda cri, inp, out: cri(out.pos_cam, inp.fg_cls_lb, inp.lb)
loss_items.cam_lb.names = ('cam_lb_fg', 'cam_lb_bg')
loss_items.cam_lb.weights = (1., 1.)

# * 开启自动混合精度。
cfg.amp.enabled = True
cfg.amp.scaler.ini.enabled = IL(lambda c: c.amp.enabled)
cfg.amp.scaler.ini.init_scale = 2.**16

# * 设定solver。
cfg.solver.max_iter = 17000  # ~25.7轮。
cfg.solver.display_step = 10
cfg.solver.loss_average_step = IL(lambda c: c.solver.display_step)
cfg.solver.save_step = IL(lambda c: max(c.solver.max_iter // 10, 1000))
cfg.solver.val_step = IL(lambda c: c.solver.save_step * 3)

# * 设定测试和验证。
def model_cfg_train2eval(c):  # noqa
    eval_model_cfg = c.model.branch_copy()
    del eval_model_cfg['initialize_seed']
    eval_model_cfg.ini.fp32 = True
    eval_model_cfg.ini.adaptive_pos_emb = True
    eval_model_cfg.resume_file = ...
    return eval_model_cfg

val_cfg = cfg.val.cfg = Config(cfgs_update_at_parser=('configs/infer_voc/square/base.py',  # noqa
                                                      'configs/infer_voc/patch_val.py'))
val_cfg.model = IL(model_cfg_train2eval, priority=10)  # 验证时，使用与训练时一样的模型。
val_cfg.io = IL(lambda c: c.io.branch_copy(), priority=10)  # 验证时，使用与训练时一样的模型IO。
val_cfg.rslt_dir = ...

infer_cfg = cfg.infer.cfg = Config(cfgs_update_at_parser=('configs/infer_voc/align/base.py',))
infer_cfg.model = IL(model_cfg_train2eval, priority=10)  # 推理时，使用与训练时一样的模型。
infer_cfg.io = IL(lambda c: c.io.branch_copy(), priority=10)  # 推理时，使用与训练时一样的模型IO。
infer_cfg.rslt_dir = ...
