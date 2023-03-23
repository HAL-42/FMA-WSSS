#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/23 12:22
@File    : scaled_ASL.py
@Software: PyCharm
@Desc    : 
"""
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .ASL import AsymmetricLoss

__all__ = ['ScaledASL']


class ScaledASL(nn.Module):

    def __init__(self,
                 s: float=1., b: float = 0., proj_lr: float | None = None, norm_s: bool=True,
                 ASL_cfg: dict | None=None
                 ):
        """将多标签分类的正类视作正样本，负类视作负样本，计算对比损失。

        Args:
            s: 缩放因子。
            b: 偏移因子。
            proj_lr: 用于学习s和b的学习率，若为None则不学习。
            norm_s: 是否对s对loss归一化，消除S对梯度的影响。
            ASL_cfg: AsymmetricLoss的配置。
        """
        super().__init__()
        self.asl = AsymmetricLoss(**(ASL_cfg if ASL_cfg is not None else {}))

        self.norm_s = norm_s
        self.proj_lr = proj_lr

        if self.learnable:
            self.s = nn.Parameter(torch.tensor(s, dtype=torch.float32))
            self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
        else:
            self.register_buffer('s', torch.tensor(s, dtype=torch.float32))
            self.register_buffer('b', torch.tensor(b, dtype=torch.float32))

    @property
    def learnable(self):
        return self.proj_lr is not None

    def get_named_param_groups(self) -> dict[str, dict]:
        """获取参数组。

        Returns:
            参数组。
        """
        if self.learnable:
            return {'ScaleASL_proj': {'params': list(self.parameters()),
                                      'lr': self.proj_lr}}  # s和b取值任意，无需weight decay。
        else:
            return dict()

    def display(self, iteration: int, writer: SummaryWriter):
        if self.learnable:
            print(f"    s: {self.s}")
            print(f"    b: {self.b}")
            writer.add_scalar(f'param/ScaledASL_s', self.s.item(), iteration + 1)
            writer.add_scalar(f'param/ScaledASL_b', self.b.item(), iteration + 1)

    def forward(self, S: torch.Tensor, cls_lb: torch.Tensor) -> torch.Tensor:
        """将多标签分类的正类视作正样本，负类视作负样本，计算对比损失。

        Args:
            S: (N, G)的相似度图。
            cls_lb: (N, G)的类别标签。

        Returns:
            多标签对比损失。
        """
        s, b = self.s.to(S.dtype), self.b.to(S.dtype)

        scaled_S = s * S + b
        asl_loss = self.asl(scaled_S, cls_lb)
        return (asl_loss / s.detach()) if self.norm_s else asl_loss
