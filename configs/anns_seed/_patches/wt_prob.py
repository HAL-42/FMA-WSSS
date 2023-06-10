#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/6/8 20:45
@File    : dp_logit.py
@Software: PyCharm
@Desc    : 
"""
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from alchemy_cat.py_tools import Config

cfg = config = Config()


def loader(cam_dir: str, img_id: str) -> (torch.Tensor, torch.Tensor):
    loaded = np.load(osp.join(cam_dir, f'{img_id}.npy'), allow_pickle=True).tolist()

    prob = torch.as_tensor(loaded['prob'], dtype=torch.float32, device=torch.cuda.current_device())  # (C, H, W)

    fg_cls = torch.as_tensor(loaded['keys'], dtype=torch.long, device=torch.cuda.current_device())  # (C,)
    if fg_cls[0] == 0:  # 如[0, 1, 15]。
        fg_cls = fg_cls[1:] - 1
    else:  # 如[1, 15]。
        F.pad(prob, (1, 0, 0, 0, 0, 0), mode='constant', value=0)  # 增加一个背景得分。
        fg_cls = fg_cls - 1

    return prob, fg_cls


cfg.cam.loader.cal = loader
