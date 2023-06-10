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


def loader(cam_dir: str, img_id: str, must_in_seed: bool=True) -> (torch.Tensor, torch.Tensor):
    logit = torch.as_tensor(np.load(osp.join(cam_dir, f'{img_id}.npy')),
                            dtype=torch.float32, device=torch.cuda.current_device())  # (C, H, W)
    prob = torch.softmax(logit, dim=0)

    if must_in_seed:
        seed = prob.argmax(dim=0)  # (H, W)
        fg_cls = torch.unique(seed, sorted=True)
        if fg_cls[0] == 0:  # 如[0, 1, 15]。
            prob = prob[fg_cls, ...]
            fg_cls = fg_cls[1:] - 1
        else:  # 如[1, 15]。
            prob = prob[[0] + fg_cls.tolist(), ...]
            fg_cls = fg_cls - 1
    else:
        fg_cls = torch.arange(0, prob.shape[0] - 1, dtype=torch.long, device=prob.device)

    return prob, fg_cls


cfg.cam.loader.ini.must_in_seed = True
cfg.cam.loader.cal = IL(lambda c: partial(loader, must_in_seed=c.cam.loader.ini.must_in_seed))
