#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/23 21:45
@File    : utils.py
@Software: PyCharm
@Desc    : 
"""
from typing import Callable

import torch

from libs.sam import SamAnns

__all__ = ['gather_anns', 'scatter_anns', 'sort_anns']


def gather_anns(arr: torch.Tensor, anns: SamAnns, gather_method: str='mean', keep_2d: bool=True) -> torch.Tensor:
    """根据anns收集arr。

    Args:
        arr: (C, H, W) 待收集张量。
        anns: SAM标注，其.data.masks为(S, H, W)的bool掩码，.data.areas为(S,)的int数组。
        gather_method: 收集方法，可选 'mean', 'sum'。
        keep_2d: 是否保留2D维度，若为True则返回(S, 1, P)的张量，否则返回(S, P)的张量。

    Returns:
        (C, 1, S) 或 (C, S) 的张量，P为标注数量。
    """
    masks = anns.data.masks

    anns_arr = arr[:, None, :, :] * masks[None, :, :, :]  # (C, 1, H, W) * (1, S, H, W) = (C, S, H, W)

    match gather_method:
        case 'mean':
            gathered_arr = torch.sum(anns_arr, dim=(2, 3)) / anns.data.areas[None, :]  # (C, S)
        case 'sum':
            gathered_arr = torch.sum(anns_arr, dim=(2, 3))  # (C, S)
        case _:
            raise ValueError(f"未知的gather_method: {gather_method}")

    if keep_2d:
        gathered_arr = gathered_arr[:, None, :]  # (C, 1, S)

    return gathered_arr


def scatter_anns(anns_vec: torch.Tensor, anns: SamAnns, default_vec=0) -> torch.Tensor:
    """将标注向量根据标注散布到各个原图上，默认标注的优先级从低到高。

    Args:
        anns_vec: ([C, ]S) 标注向量，S为标注数量。
        anns: SAM标注，其.data.masks为(S, H, W)的bool掩码。
        default_vec: 默认标注向量。若为向量，形状应与(C,)，若为标量，则将其广播到anns_vec的形状。

    Returns:
        ([C, ]H, W) 张量
    """
    with_C = anns_vec.ndim == 2
    if not with_C:
        anns_vec = anns_vec[None, :]

    masks = anns.data.masks

    if torch.is_tensor(default_vec):
        assert default_vec.dtype == anns_vec.dtype
        assert default_vec.device == anns_vec.device
        assert default_vec.shape == anns_vec.shape[:1]
        default_vec = default_vec[:, None, None]

    scattered_arr = torch.zeros((anns_vec.shape[0], *masks.shape[1:]), dtype=anns_vec.dtype, device=anns_vec.device)
    scattered_arr += default_vec

    # NOTE 若此处有性能瓶颈，则可用argmax(masks, dim=0)+无ann位置为S+整数索引(anns_vec, default)代替。
    for m, v in zip(masks, anns_vec.T, strict=True):  # (H, W), (C,) in (S, H, W), (S, C)
        scattered_arr[:, m] = v

    if not with_C:
        scattered_arr = scattered_arr[0, :, :]

    return scattered_arr


def sort_anns(anns: SamAnns,
              priority: Callable[[dict[str, ...]], int | tuple] | str | tuple[str]='area_smaller') -> SamAnns:
    """对标注按照priority排序。

    Args:
        anns: 待排序的标注。
        priority: 排序的priority，输入标注，返回值用于指定标注的优先级。

    Returns:
        排序后的anns，优先即从低到高。
    """
    if isinstance(priority, str):
        priority = (priority,)

    if isinstance(priority, tuple):

        def pri(ann: dict[str, ...]) -> list[int]:
            ret = []

            for p in priority:
                match p:
                    case 'area_bigger':
                        ret.append(ann['area'])
                    case 'area_smaller':
                        ret.append(-ann['area'])
                    case 'level_bigger':
                        ret.append(ann['level'])
                    case 'level_smaller':
                        ret.append(-ann['level'])
                    # NOTE 先看面积再看level无意义，因为面积很少会一样。
                    case 'conf_bigger':
                        ret.append(ann['conf'])
                    case 'conf_smaller':  # 这句应该没用。
                        ret.append(-ann['conf'])
                    case _:
                        raise ValueError(f"未知的priority: {p}")

            return ret
    else:
        pri = priority

    sorted_anns = SamAnns(sorted(anns, key=pri, reverse=False))
    sorted_anns.stack_data_like(anns)

    return sorted_anns
