#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/28 12:18
@File    : caller.py
@Software: PyCharm
@Desc    : 
"""
from typing import Optional

import torch

from .tools import att2aff, get_aff_mask, aff_cam
from ..score import cam2score_cuda


def att_cam(att: torch.Tensor, cam: torch.Tensor,
            att2aff_cfg: dict, aff_mask_cfg: dict, aff_cfg: dict,
            aff_at: str, dsize: Optional[None]=None) -> torch.Tensor:
    """用att_weight优化CAM。

    Args:
        att: 未合并的D(HW+1)(HW+1)，或合并后的(HW, HW)的attention矩阵。
        cam: PHW的CAM。
        att2aff_cfg: att2aff的配置参数。
        aff_mask_cfg: get_aff_mask的配置参数。
        aff_cfg: aff_cam的配置参数。
        aff_at: 对谁使用aff优化，'score' 或 'cam'。
        dsize: 将CAM和att缩放到dsize后优化。

    Returns:
        优化后的CAM（注意，哪怕是aff_at score，优化后也不符合score的min=0，max=1格式）。使用前可能要重新归一化。
    """
    # * 缩放CAM和att。
    if dsize is not None:
        raise NotImplementedError

    # * 获取score。
    score = cam2score_cuda(cam, cam.shape[1:], resize_first=True)  # PHW

    # * aff次数为0时，那就什么都不做。
    if aff_cfg['n_iters'] == 0:
        match aff_at:
            case 'score':
                return score
            case 'cam':
                return cam
            case _:
                raise ValueError(f"aff_at must be 'score' or 'cam', but got {aff_at}")

    # * 获取affinity。
    aff = att2aff(att, **att2aff_cfg)

    # * 获取aff_mask。
    aff_mask = get_aff_mask(score.detach(), **aff_mask_cfg)

    # * 将aff作用到CAM上。
    match aff_at:
        case 'score':
            to_aff = score
        case 'cam':
            to_aff = cam
        case _:
            raise ValueError(f"aff_at must be 'score' or 'cam', but got {aff_at}")
    cam_affed = aff_cam(aff, aff_mask, to_aff, **aff_cfg)

    return cam_affed
