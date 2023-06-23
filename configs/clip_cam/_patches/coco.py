#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/6/22 3:33
@File    : coco.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config, IL

from libs.data import COCO

cfg = config = Config('configs/patterns/coco_names/clip_es.py',
                      'configs/patterns/aug/coco_rand_range.py')

# * 设定数据集。
cfg.dt.train.set_whole(True)
# ** 设定训练集。
cfg.dt.train.ini.cls_labels_type = 'seg_cls_labels'
cfg.dt.train.ini.ps_mask_dir = None
cfg.dt.train.dt = IL(lambda c:
                     COCO(root='datasets', split='train', **c.dt.train.ini),
                     priority=-10)

cfg.dt.train.epoch_len = IL(lambda c:
                            len(c.dt.train.dt) // c.loader.train.batch_size,
                            priority=0)

# * 设定增强器的scale_crop。
cfg.auger.train.ini.scale_crop_method.crop_size = 272
cfg.auger.train.ini.scale_crop_method.high_low_ratio = 4 / 3

# * 设定数据管理器。
cfg.loader.train.sub_iter_num = 1  # 由于输入图像尺寸变大、pos_cam变多，需要更多的sub_iter。

# * 设定Scheduler。
cfg.sched.warm.warm_iters = 1000  # ~0.2个epoch。

# * 设定solver。
cfg.solver.max_iter = 34000  # ~6.8个epoch

# * 设定测试和验证。
def model_cfg_train2eval(c):  # noqa
    eval_model_cfg = c.model.branch_copy()
    del eval_model_cfg['initialize_seed']
    eval_model_cfg.ini.fp32 = True
    eval_model_cfg.ini.adaptive_pos_emb = True
    eval_model_cfg.resume_file = ...
    return eval_model_cfg

cfg.val.set_whole(True)
cfg.val.cfg = Config(cfgs_update_at_parser=('configs/infer_voc/square/base_coco.py',  # noqa
                                            'configs/infer_voc/patch_val_coco.py'))
cfg.val.cfg.model = IL(model_cfg_train2eval, priority=10)  # 验证时，使用与训练时一样的模型。
cfg.val.cfg.io = IL(lambda c: c.io.branch_copy(), priority=10)  # 验证时，使用与训练时一样的模型IO。
cfg.val.cfg.rslt_dir = ...

cfg.infer.set_whole(True)
cfg.infer.cfg = Config(cfgs_update_at_parser=('configs/infer_voc/base_coco.py',
                                              'configs/infer_voc/patch_train_infer_coco.py'))
cfg.infer.cfg.model = IL(model_cfg_train2eval, priority=10)  # 推理时，使用与训练时一样的模型。
cfg.infer.cfg.io = IL(lambda c: c.io.branch_copy(), priority=10)  # 推理时，使用与训练时一样的模型IO。
cfg.infer.cfg.rslt_dir = ...
