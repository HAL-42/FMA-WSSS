#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/16 15:37
@File    : score.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np
from alchemy_cat.alg import size2HW

from utils.norm import min_max_norm
from utils.resize import resize_cam


def cam2score(cam: np.ndarray, dsize, resize_first: bool) -> np.ndarray:
    h, w = size2HW(dsize)
    if resize_first:
        score = resize_cam(cam, (h, w))
        score = min_max_norm(score, dim=(1, 2), thresh=0.)
    else:
        score = min_max_norm(cam, dim=(1, 2), thresh=0.)
        score = resize_cam(score, (h, w))

    return score
