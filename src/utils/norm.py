#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/9 23:25
@File    : norm.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np
import torch


def min_max_norm(arr: torch.Tensor, dim: int | tuple, detach_min_max: bool=True) -> torch.Tensor:
    """在指定维度min-max归一化输入张量。

    Args:
        arr: 输入张量。
        dim: 做归一化的维度。
        detach_min_max: 是否分离min和max。

    Returns:
        归一化后的张量。
    """
    # * 若为numpy数组，转为torch张量。
    if isinstance(arr, np.ndarray):
        is_numpy = True
        arr = torch.from_numpy(arr)

    arr_min, arr_max = arr.amin(dim=dim, keepdim=True), arr.amax(dim=dim, keepdim=True)
    if detach_min_max:
        arr_min, arr_max = arr_min.detach(), arr_max.detach()
    arr_normed = (arr - arr_min) / (arr_max - arr_min)

    # * 若为numpy数组，转为numpy数组。
    if is_numpy:
        arr_normed = arr_normed.numpy()

    return arr_normed
