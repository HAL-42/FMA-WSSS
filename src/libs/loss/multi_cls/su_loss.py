#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/22 16:58
@File    : sn_sp.py
@Software: PyCharm
@Desc    : 
"""
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class MultiLabelSuLoss(nn.Module):

    def __init__(self,
                 thresh: float | None = None,
                 thresh_lr: float | None = None,
                 reduce: str = 'mean'):
        super().__init__()
        self.thresh_lr = thresh_lr
        self.reduce = reduce

        if thresh is None:
            self.thresh = None
        elif thresh_lr is not None:
            self.thresh = nn.Parameter(torch.tensor(thresh, dtype=torch.float32))
        else:
            self.register_buffer('thresh', torch.tensor(thresh, dtype=torch.float32))

    @property
    def learnable(self) -> bool:
        return (self.thresh is not None) and (self.thresh_lr is not None)

    def get_named_param_groups(self) -> dict[str, dict]:
        """获取参数组。

        Returns:
            参数组。
        """
        if self.learnable:
            return {'MultiLabelSuLoss_thresh': {'params': list(self.parameters()),
                                                'lr': self.thresh_lr}}  # thresh取值任意，无需weight decay。
        else:
            return dict()

    def display(self, iteration: int, writer: SummaryWriter):
        if self.learnable:
            print(f"    thresh: {self.thresh}")
            writer.add_scalar(f'param/MultiLabelSuLoss_thresh', self.thresh.item(), iteration + 1)

    def forward(self, S: torch.Tensor, cls_lb: torch.Tensor) -> torch.Tensor:
        """计算可能带有阈值的苏剑林多分类损失。

        Args:
            S: (N, G)的logits。
            cls_lb: (N, G)的one-hot标签。

        Returns:
            基于reduction合并后的损失。
        """
        neg_p_S = (1 - 2 * cls_lb) * S  # (N, G)，正类的S取负，负类的S不变。
        e_neg_p_S = torch.exp(neg_p_S)  # (N, G)的e^{neg_p_S}
        Sigma_e_Sn = (e_neg_p_S * (cls_lb == 0)).sum(dim=1)  # (N,)，负类的e^{Sn}求和。
        sigma_e_neg_Sp = (e_neg_p_S * (cls_lb == 1)).sum(dim=1)  # (N,)，正类的e^{-Sp}求和。

        if self.thresh is not None:
            thresh = self.thresh.to(S.dtype)
            e_thresh, e_neg_thresh = torch.exp(thresh), torch.exp(-thresh)

            neg_thresh_Sigma_e_Sn = e_neg_thresh * Sigma_e_Sn  # (N,)，负类的e^{-thresh} * Sigma_e_Sn。
            thresh_Sigma_e_neg_Sp = e_thresh * sigma_e_neg_Sp  # (N,)，正类的e^{thresh} * sigma_e_neg_Sp。
        else:
            neg_thresh_Sigma_e_Sn = 0
            thresh_Sigma_e_neg_Sp = 0

        loss = torch.log(1 + Sigma_e_Sn * sigma_e_neg_Sp + neg_thresh_Sigma_e_Sn + thresh_Sigma_e_neg_Sp)  # (N,)

        if self.reduce == "none":
            ret = loss
        elif self.reduce == "mean":
            ret = loss.mean()
        elif self.reduce == "sum":
            ret = loss.sum()
        else:
            raise ValueError(self.reduce + " is not valid")
        return ret


class MultiLabelSuThreshLoss(nn.Module):

    def __init__(self,
                 reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """官方的苏剑林多分类损失。"""
        y_pred = (1 - 2 * y_true) * y_pred

        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12

        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)

        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        loss = neg_loss + pos_loss

        if self.reduction == "none":
            ret = loss
        elif self.reduction == "mean":
            ret = loss.mean()
        elif self.reduction == "sum":
            ret = loss.sum()
        else:
            raise ValueError(self.reduction + " is not valid")
        return ret
