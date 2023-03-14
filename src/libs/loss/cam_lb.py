#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/9 23:21
@File    : l1_cam.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn.functional as F
from torch import nn


class CAMIntensityLoss(nn.Module):

    def __init__(self,
                 loss_type: str = 'l1', reduce: str = 'all', detach_max: bool = True, bg_thresh: float = 0.,
                 eps: float = 1e-5):
        super().__init__()
        self.loss_type = loss_type  # l1, l2
        self.reduce = reduce  # all, fg-bg, fg-bg-per-pos
        self.detach_max = detach_max
        self.bg_thresh = bg_thresh
        self.eps = eps

    def forward(self, pos_cam: torch.Tensor, fg_cls_lb: torch.Tensor, lb: torch.Tensor)\
            -> tuple[torch.Tensor, torch.Tensor]:
        if pos_cam.shape[1:] != lb.shape[1:]:
            pos_cam = F.interpolate(pos_cam[:, None, :, :], size=lb.shape[1:],
                                    mode='bilinear', align_corners=False)[:, 0, :, :]

        pos_bt_idx, pos_fg_cls = torch.nonzero(fg_cls_lb, as_tuple=True)
        pos_lb = lb[pos_bt_idx, :, :]  # NHW -> PHW

        fg_mask = (pos_lb == (pos_fg_cls + 1)[:, None, None])  # 转为PHW的0，1掩码。
        bg_mask = (~fg_mask & (pos_lb != 255))  # 背景掩码，非前景且非忽略区域。
        fg_mask = fg_mask.to(pos_cam.dtype)  # 转为浮点数。
        bg_mask = bg_mask.to(pos_cam.dtype)

        fg_nums = fg_mask.sum(dim=(1, 2))  # P
        bg_nums = bg_mask.sum(dim=(1, 2))  # P

        fg_cam = pos_cam * fg_mask  # PHW
        bg_cam = pos_cam * bg_mask  # PHW

        max_fg_cam = torch.amax(fg_cam, dim=(1, 2), keepdim=True)
        max_fg_cam = torch.maximum(max_fg_cam, torch.tensor(0,                       # Px1x1，各正类前景上最大值，至少是0。
                                                            dtype=max_fg_cam.dtype,
                                                            device=max_fg_cam.device))
        if self.detach_max:
            max_fg_cam = max_fg_cam.detach()  # 如不detach，每个前景像素都会要求减小最大CAM。
        fg_l1 = max_fg_cam - fg_cam  # PHW，前背景都有。

        bg_l1 = torch.maximum(bg_cam, torch.tensor(self.bg_thresh,
                                                   dtype=bg_cam.dtype,
                                                   device=bg_cam.device)) - self.bg_thresh

        match self.loss_type:
            case 'l1':
                fg_l, bg_l = torch.abs(fg_l1), torch.abs(bg_l1)
            case 'l2':
                fg_l, bg_l = 0.5 * fg_l1 ** 2, 0.5 * bg_l1 ** 2
            case _:
                raise ValueError(f"不支持的{self.loss_type=}")

        fg_l = fg_l * fg_mask  # 背景上loss归零（虽然背景上loss不会回传，但是影响具体数值）
        bg_l = bg_l * bg_mask

        match self.reduce:
            case 'all':
                all_nums = fg_nums.sum() + bg_nums.sum()
                fg_loss, bg_loss = fg_l.sum() / all_nums, bg_l.sum() / all_nums  # /PHW
            case 'fg-bg':
                fg_loss = 0.5 * fg_l.sum() / (fg_nums.sum() + self.eps)
                bg_loss = 0.5 * bg_l.sum() / (bg_nums.sum() + self.eps)
            case 'fg-bg-per-pos':
                fg_loss_per_pos = 0.5 * fg_l.sum(dim=(1, 2)) / (fg_nums + self.eps)  # P
                bg_loss_per_pos = 0.5 * bg_l.sum(dim=(1, 2)) / (bg_nums + self.eps)
                fg_loss, bg_loss = fg_loss_per_pos.mean(), bg_loss_per_pos.mean()
            case _:
                raise ValueError(f"不支持的{self.reduce=}")

        return fg_loss, bg_loss
