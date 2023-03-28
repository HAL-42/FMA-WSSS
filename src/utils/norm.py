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


def min_max_norm(arr: torch.Tensor | np.ndarray, dim: int | tuple, detach_min_max: bool=True,
                 thresh: float | None =None, eps: float=1e-7) -> torch.Tensor:
    """在指定维度min-max归一化输入张量。

    Args:
        arr: 输入张量。
        dim: 做归一化的维度。
        detach_min_max: 是否分离min和max。
        thresh: 若不为None，将归一化前张量中小于thresh的值置为thresh。
        eps: 防止除0错误的极小值。

    Returns:
        归一化后的张量。
    """
    # * 若为numpy数组，转为torch张量。
    if is_numpy := isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr)

    # * 应用阈值。
    if thresh is not None:
        arr = torch.maximum(arr, torch.tensor(thresh, device=arr.device, dtype=arr.dtype))

    # * 归一化。
    arr_min, arr_max = arr.amin(dim=dim, keepdim=True), arr.amax(dim=dim, keepdim=True)
    if detach_min_max:
        arr_min, arr_max = arr_min.detach(), arr_max.detach()
    arr_normed = (arr - arr_min) / (arr_max - arr_min + eps)  # 其实用max(max - min, 1e-7)更安全（已经归一化过的不变）。

    # * 若为numpy数组，转为numpy数组。
    if is_numpy:
        arr_normed = arr_normed.numpy()

    return arr_normed
