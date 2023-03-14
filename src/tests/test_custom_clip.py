#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/8 16:52
@File    : test_custom_clip.py
@Software: PyCharm
@Desc    : 
"""
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from libs.data import VOCAug2, VOC2Auger, FewShotDt
from libs.clip import load
from libs.coop.custom_clips.cam_clips import GradCAMCLIP


bg_names = ['ground', 'land', 'grass', 'tree', 'building',
            'wall', 'sky', 'lake', 'water', 'river', 'sea',
            'railway', 'railroad', 'keyboard', 'helmet', 'cloud',
            'house', 'mountain', 'ocean', 'road', 'rock',
            'street', 'valley', 'bridge', 'sign',
            ]
class_names = ['aeroplane', 'bicycle', 'bird avian', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair seat', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person with clothes,people,human',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor screen',
               ]


def test_GradCAMCLIP():
    dt = VOCAug2(root='datasets', split='train_aug')
    dt = FewShotDt(dt, 16, seed=10, except_bg=False)
    auger = VOC2Auger(dt, scale_crop_method={'method': 'rand_range',
                                             'low_size': 224, 'high_size': 368, 'short_thresh': 224,
                                             'crop_size': 224},
                      lb_scale_factor=16)
    inp = auger[0]
    img_ = inp.img
    img = inp.img[None, ...].to('cuda:0')
    ol_cls_lb_ = inp.ol_cls_lb
    ol_cls_lb = torch.from_numpy(ol_cls_lb_)[None, ...].to('cuda:0')

    clip_model, _ = load('ViT-B/16', device='cpu', jit=False, cpu_float=True)
    gcam_clip = GradCAMCLIP(clip_model, class_names + bg_names,
                            {'n_ctx': 16, 'ctx_init': "a clean origami", 'csc': False, 'cls_token_pos': 'end'},
                            adaptive_pos_emb=False, sm_fg_exist=True)
    gcam_clip.set_mode('train')
    gcam_clip.to('cuda:0')

    out = gcam_clip(img, ol_cls_lb[:, 1:])

    pos_gcam = out.pos_cam
    hd_pos_gcam = F.interpolate(pos_gcam[None, ...], (224, 224), mode='bilinear')  # TODO: align

    loss = -hd_pos_gcam.mean()
    loss.backward()

    hd_pos_gcam_ = hd_pos_gcam.detach().cpu().numpy()[0]
    PIL_img = auger.inv2PIL(img_)

    fig: plt.Figure = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(PIL_img)
    ax.imshow(hd_pos_gcam_[0], cmap='jet', alpha=0.5)

    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(PIL_img)
    ax.imshow(hd_pos_gcam_[1], cmap='jet', alpha=0.5)

    fig.show()
    plt.show()
