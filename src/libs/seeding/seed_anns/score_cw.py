#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/24 22:11
@File    : score_cw.py
@Software: PyCharm
@Desc    : 
"""
import torch

from libs.sam import SamAnns
from .utils import gather_anns, scatter_anns, sort_anns
from ..score import cam2score_cuda, cat_bg_score_cuda, idx2seed_cuda


def gather_norm_bg_argmax(anns: SamAnns, cam: torch.Tensor, fg_cls: torch.Tensor,
                          norm_first: bool,
                          gather_method: str,
                          bg_method: dict,
                          priority: str | tuple[str, ...]) -> torch.Tensor:
    """先收集标注的CAM响应，再在标注间做归一化，计算背景得分，channel维度上argmax后得到种子点。

    Args:
        anns: SAM标注，应该已经完成stack_data。
        cam: (P, H, W) 的CAM响应。
        fg_cls: (P,) 的前景类别。
        norm_first: 是否先归一化再计算标注得分。
        gather_method: 收集配置。
        bg_method: 背景分计算配置。
        priority: 标注排序优先级。

    Returns:
        (H, W) 种子点。
    """
    if norm_first:
        # * 截0，归一化，计算前景得分。
        fg_score = cam2score_cuda(cam, dsize=None, resize_first=True)  # (C, 1, S)
        # 等价形式为：
        # fg_score = torch.maximum(cam, torch.tensor(0, device=cam.device, dtype=cam.dtype))

        # * 收集标注上的前景得分。
        anns_fg_score = gather_anns(fg_score, anns, keep_2d=True, gather_method=gather_method)  # (C, 1, S)

        # * 再次归一化。
        anns_fg_score = cam2score_cuda(anns_fg_score, dsize=None, resize_first=True)  # (C, 1, S)
    else:
        # * 收集标注上的CAM响应。
        anns_cam = gather_anns(cam, anns, keep_2d=True, gather_method=gather_method)  # (C, 1, S)

        # * 截0，归一化，计算标注前景得分。
        anns_fg_score = cam2score_cuda(anns_cam, dsize=None, resize_first=True)  # (C, 1, S)

    # * 计算标注背景得分。
    anns_score = cat_bg_score_cuda(anns_fg_score, bg_method)  # (C+1, 1, S)

    anns_max_idx = torch.argmax(anns_score, dim=0)  # (1, S)

    # * 得到标注种子点。
    anns_seed = idx2seed_cuda(anns_max_idx, fg_cls)  # (1, S)
    anns.add_item('seed', anns_seed[0].tolist())

    # * 以得分为标注置信度。
    # anns_conf = torch.gather(anns_score, dim=0, index=anns_max_idx[None, ...])  # (1, S)，与max效果相同。
    anns_conf = anns_score[anns_max_idx,
                           torch.arange(anns_score.shape[1])[:, None],
                           torch.arange(anns_score.shape[2])[None, :]]  # (1, S)，与max效果相同。
    anns.add_item('conf', anns_conf[0].tolist())

    # * 对标注排序。
    sorted_anns = sort_anns(anns, priority=priority)

    # * 得到最终种子点。
    sorted_anns_seed = torch.as_tensor([ann['seed'] for ann in sorted_anns],
                                       dtype=anns_seed.dtype, device=anns_seed.device)  # (S,)
    seed = scatter_anns(sorted_anns_seed, sorted_anns, default_vec=0)

    return seed
