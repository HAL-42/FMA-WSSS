#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/22 22:06
@File    : sharpness.py
@Software: PyCharm
@Desc    : 
"""
import torch
from torch import nn


class MultiLabelSharpness(nn.Module):

    def __init__(self,
                 reduce: str = 'batch_mean',
                 loss_type: str='sharpness',
                 correction: int=0,
                 eps: float=1e-6):
        """按照CLIP-ES的公式(5)，计算正类Logit的尖锐度。

        Args:
            reduce: 可以为：mean，在batch维度平均每个样本的尖锐度；batch_mean: 完全按照CLIP-ES的公式(5)，
                在整个batch上计算尖锐度。
            loss_type: 可以为：sharpness，计算正类Logit的尖锐度；
            correction: 计算方差/标准差时的修正项。
        """
        super().__init__()
        self.reduce = reduce
        self.loss_type = loss_type
        self.correction = correction
        self.eps = eps

    def forward(self, S: torch.Tensor, cls_lb: torch.Tensor) -> torch.Tensor:
        """将多标签分类的正类视作正样本，负类视作负样本，计算对比损失。

        Args:
            S: (N, G)的相似度图/概率。
            cls_lb: (N, G)的类别标签。

        Returns:
            正类Logit的尖锐度。
        """
        pos_mask = cls_lb == 1

        valid_mask = (pos_num := pos_mask.sum(dim=1, dtype=torch.int32)) >= 2
        if torch.all(~valid_mask):  # 无正样本
            return 0 * S.sum()
        valid_pos_mask = pos_mask[valid_mask, :]
        valid_pos_num = pos_num[valid_mask]
        valid_S = S[valid_mask, :]

        mean = (valid_S * valid_pos_mask).sum(dim=1) / valid_pos_num

        var = ((valid_S - mean[:, None]) ** 2 * valid_pos_mask).sum(dim=1)
        var = var / (valid_pos_num - self.correction)

        match self.loss_type:
            case 'sharpness':
                n = var
                d = mean
            case 'CV':  # 变异系数
                std = torch.sqrt(var)
                n = std
                d = mean
            case 'CV^2':
                n = var
                d = mean ** 2
            case _:
                raise ValueError(f'Unknown {self.loss_type=}')

        match self.reduce:
            case 'mean':
                return (n / (d + self.eps)).mean()
            case 'batch_mean':
                return n.sum() / (d.sum() + self.eps)
