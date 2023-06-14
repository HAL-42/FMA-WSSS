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
from functools import partial

import numpy as np
import torch
from alchemy_cat.py_tools import Config, IL

cfg = config = Config()


def loader(cam_dir: str, img_id: str, score_type: str='prob') -> (torch.Tensor, torch.Tensor):
    loaded = np.load(osp.join(cam_dir, f'{img_id}.npz'))

    prob = torch.as_tensor(loaded[score_type], dtype=torch.float32, device=torch.cuda.current_device())  # (C, H, W)

    fg_cls = torch.as_tensor(loaded['fg_cls'], dtype=torch.long, device=torch.cuda.current_device())  # (C,)

    return prob, fg_cls


cfg.cam.loader.ini.score_type = 'prob'
cfg.cam.loader.cal = IL(lambda c: partial(loader, **c.cam.loader.ini))
