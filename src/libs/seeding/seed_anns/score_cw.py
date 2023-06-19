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
                          norm_first: bool | str,
                          gather_method: str,
                          bg_method: dict,
                          priority: str | tuple[str, ...],
                          ret_seeded_anns: bool=False) -> torch.Tensor | tuple[torch.Tensor, SamAnns]:
    """先收集标注的CAM响应，再在标注间做归一化，计算背景得分，channel维度上argmax后得到种子点。

    Args:
        anns: SAM标注，应该已经完成stack_data。
        cam: (P, H, W) 的CAM响应。
        fg_cls: (P,) 的前景类别。
        norm_first: 是否先归一化再计算标注得分。
        gather_method: 收集配置。
        bg_method: 背景分计算配置。
        priority: 标注排序优先级。
        ret_seeded_anns: 是否返回排序后，带有置信度得分和种子的。

    Returns:
        (H, W) 种子点。
    """
    match norm_first:
        case True:
            # * 截0，计算前景得分。
            fg_score = torch.maximum(cam, torch.tensor(0, device=cam.device, dtype=cam.dtype))  # (C, 1, S)

            # * 收集标注上的前景得分。
            anns_fg_score = gather_anns(fg_score, anns, keep_2d=True, gather_method=gather_method)  # (C, 1, S)

            # * 再次归一化。
            anns_fg_score = cam2score_cuda(anns_fg_score, dsize=None, resize_first=True)  # (C, 1, S)
        case False:
            # * 收集标注上的CAM响应。
            anns_cam = gather_anns(cam, anns, keep_2d=True, gather_method=gather_method)  # (C, 1, S)

            # * 截0，归一化，计算标注前景得分。
            anns_fg_score = cam2score_cuda(anns_cam, dsize=None, resize_first=True)  # (C, 1, S)
        case 'double_norm':
            # * 截0，归一化，计算前景得分。
            fg_score = cam2score_cuda(cam, dsize=None, resize_first=True)  # (C, 1, S)

            # * 收集标注上的前景得分。
            anns_fg_score = gather_anns(fg_score, anns, keep_2d=True, gather_method=gather_method)  # (C, 1, S)

            # * 再次归一化。
            anns_fg_score = cam2score_cuda(anns_fg_score, dsize=None, resize_first=True)  # (C, 1, S)
        case 'no_norm':
            # * 收集标注上的得分。
            anns_fg_score = gather_anns(cam, anns, keep_2d=True, gather_method=gather_method)  # (C, 1, S)
        case {'method': 'gamma_softmax', 'gamma': gamma}:
            # * 得分乘以gamma后，做softmax归一化。
            fg_score = torch.softmax(cam * gamma, dim=0)  # (C, H, W)

            # * 收集标注上得分。
            anns_fg_score = gather_anns(fg_score, anns, keep_2d=True, gather_method=gather_method)
        case _:
            raise ValueError(f"norm_first should be bool or 'double_norm', got {norm_first}")

    # * 计算标注背景得分。
    anns_score = cat_bg_score_cuda(anns_fg_score, bg_method)  # (C+1, 1, S)
    if ret_seeded_anns:
        anns.add_item('score', anns_score[:, 0, :].T)

    anns_max_idx = torch.argmax(anns_score, dim=0)  # (1, S)

    # * 得到标注种子点。
    anns_seed = idx2seed_cuda(anns_max_idx, fg_cls)  # (1, S)
    anns.add_item('seed', anns_seed[0].tolist())

    # * 以得分为标注置信度。
    anns_conf = torch.gather(anns_score, dim=0, index=anns_max_idx[None, ...])[0]  # (1, S)，与max效果相同。
    # anns_conf = anns_score[anns_max_idx,
    #                        torch.arange(anns_score.shape[1])[:, None],
    #                        torch.arange(anns_score.shape[2])[None, :]]  # (1, S)，与max效果相同。
    anns.add_item('conf', anns_conf[0].tolist())

    # * 对标注排序。
    sorted_anns = sort_anns(anns, priority=priority)

    # * 得到最终种子点。
    sorted_anns_seed = torch.as_tensor([ann['seed'] for ann in sorted_anns],
                                       dtype=anns_seed.dtype, device=anns_seed.device)  # (S,)
    seed = scatter_anns(sorted_anns_seed, sorted_anns, default_vec=0)

    if not ret_seeded_anns:
        return seed
    else:
        return seed, sorted_anns
