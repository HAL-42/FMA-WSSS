#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/16 14:35
@File    : resize_cams.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np
from cv2 import cv2

from alchemy_cat.alg import size2HW


def resize_cam(cam: np.ndarray, size, interpolation: int=cv2.INTER_LINEAR) -> np.ndarray:
    """将cam采样到指定尺寸。

    Args:
        cam: (N, ori_h, ori_w)的cam。
        size: (h, w)的尺寸，若为int，则为H，W相同。
        interpolation: 插值方式。

    Returns:
        (N, H, W)的cam。
    """
    ori_h, ori_w = cam.shape[1:]
    h, w = size2HW(size)

    if (ori_h, ori_w) != (h, w):
        cam = cv2.resize(cam.transpose(1, 2, 0), (w, h), interpolation=interpolation)
        if cam.ndim == 2:  # N=1时，cv2.resize会丢掉维度。
            cam = cam[None, :, :]
        else:
            cam = cam.transpose(2, 0, 1)

    return cam
