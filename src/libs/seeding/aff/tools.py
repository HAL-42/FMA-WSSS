#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/27 16:29
@File    : tools.py
@Software: PyCharm
@Desc    : 
"""
import torch

from .utils import scoremap2bbox


def merge_att(att: torch.Tensor, last_n_layers: int) -> torch.Tensor:
    """将多个attention矩阵合并为一个。

    Args:
        att: D(HW+1)(HW+1)的attention矩阵。
        last_n_layers: 只平均最后几层的attention。

    Returns:
        (HW, HW)的attention矩阵。
    """
    att = att[-last_n_layers:, 1:, 1:]  # (last_n_layers, hw, hw)
    return torch.mean(att, dim=0)  # (hw, hw)，平均最后n个layer的att weight。


def att2aff(att: torch.Tensor, last_n_layers: int, method: dict) -> torch.Tensor:
    """将attention矩阵转化为affinity矩阵。

    Args:
        att: 未合并的D(HW+1)(HW+1)，或合并后的(HW, HW)的attention矩阵。
        last_n_layers: 只平均最后几层的attention。
        method: 描述affinity的归一化方法。

    Returns:
        (HW, HW)的affinity矩阵。
    """
    if att.ndim == 3:
        att = merge_att(att, last_n_layers)  # (hw, hw)，平均最后n个layer的att weight。

    match method:
        case {'type': 'sink-horn', 'n_iter': int(n_iter)}:
            aff = att
            for _ in range(n_iter):
                aff = aff / torch.sum(aff, dim=0, keepdim=True)
                aff = aff / torch.sum(aff, dim=1, keepdim=True)
            aff = (aff + aff.transpose(1, 0)) / 2  # sink-horn + 对称平均，得到affinity。
        case _:
            raise ValueError(f"Unknown method {method}")

    return aff


def get_aff_mask(score: torch.Tensor, method: dict) -> torch.Tensor:
    """获取affinity mask。

    Args:
        score: (P, H, W)的score。
        method: 描述affinity mask的生成方法。

    Returns:
        (P, 1或HW，1或HW)的affinity mask。
    """
    P, H, W = score.shape

    match method:
        case {'type': 'thresh-bbox', 'thresh': thresh, 'to_in_bbox': to_in_bbox, 'to_out_bbox': to_out_bbox}:
            aff_mask = torch.zeros_like(score)  # (P, H, W)

            for s, m in zip(score.detach().cpu().numpy(), aff_mask):  # 遍历每个样本的scoremap。
                box, _ = scoremap2bbox(scoremap=s, threshold=thresh, multi_contour_eval=True)
                for b in box:  # BBox内区域，才被聚合。
                    x0, y0, x1, y1 = b
                    m[y0:y1, x0:x1] = 1

            aff_mask = aff_mask.view(P, H * W)

            def aff_from(mask: torch.Tensor, fr: str) -> torch.Tensor:
                mask = mask.view(mask.shape[0], 1, -1)  # (P, 1, HW)
                if fr == 'in_bbox':
                    return mask
                elif fr == 'out_bbox':
                    return 1 - mask
                elif fr == 'all':
                    return torch.ones_like(mask)
                else:
                    raise ValueError(f"Unknown from {fr}")

            aff_mask_in_bbox = aff_mask.view(P, H * W, 1) * aff_from(aff_mask, to_in_bbox)  # (P, HW, HW)
            aff_mask_out_bbox = (1 - aff_mask.view(P, H * W, 1)) * aff_from(aff_mask, to_out_bbox)  # (P, HW, HW)

            aff_mask = aff_mask_in_bbox + aff_mask_out_bbox
        case {'type': 'all'}:
            aff_mask = torch.ones_like(score[:, 0, 0]).view(P, 1, 1)
        case _:
            raise ValueError(f"Unknown method {method}")

    return aff_mask


def aff_cam(aff: torch.Tensor, aff_mask: torch.Tensor, cam: torch.Tensor, n_iters: int) -> torch.Tensor:
    """用affinity聚合优化cam。

    Args:
        aff: (HW, HW)的affinity矩阵。
        aff_mask: (P, 1或HW，1或HW)的affinity mask。
        cam: (P, H, W)cam。
        n_iters: cam聚合的次数。

    Returns:
        (P, H, W)聚合后的cam。
    """
    P, H, W = cam.shape

    aff = torch.linalg.matrix_power(aff, n_iters)  # (HW, HW)
    aff = aff[None, :, :] * aff_mask  # (P, HW, HW)

    cam = cam.view(P, H * W, 1)

    # (P, HW, HW) @ (P, HW, 1)->(n,1)
    cam_affed = (aff @ cam).view(P, H, W)

    return cam_affed
