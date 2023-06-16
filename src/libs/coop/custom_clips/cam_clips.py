#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/7 21:30
@File    : cam_clips.py
@Software: PyCharm
@Desc    : 
"""
from typing import Any

from addict import Dict

import torch
from torch import nn
from torch.autograd import grad

from alchemy_cat.alg import MaskedSoftmax

from libs.clip.model import CLIP
from .. import prompt_learners as pl
from .. import text_encoders as te
from .. import image_encoders as ie


class GradCAMCLIP(nn.Module):
    def __init__(self, clip_model: CLIP,
                 classnames: list[str], ctx_cfg: dict,
                 adaptive_pos_emb: bool=False,
                 sm_fg_exist: bool=True):
        super().__init__()
        self.sm_fg_exist = sm_fg_exist

        self.prompt_learner = pl.CoOpLearner(classnames, clip_model, **ctx_cfg)  # 根据类名，提前emb好上下文可学的prompt。
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts  # 编码后（不带上下文）的prompt。

        self.text_encoder = te.EncEmb(clip_model)  # 文本编码器改为接收牌和牌嵌入。
        # 图像编码器返回ln1、注意力图和多个投影结果。
        self.image_encoder = ie.GetLN1(clip_model, adaptive_pos_emb=adaptive_pos_emb)

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.softmax = MaskedSoftmax(dim=-1)

        self._mode = 'train'
        self.set_mode(self.mode)

    def initialize(self, seed: Any):
        self.prompt_learner.initialize(seed)

    def get_logits(self, img: torch.Tensor, pad_info: dict[str, ...]=None) -> Dict:
        # * 获取文本特征。
        prompts = self.prompt_learner().to(self.dtype)  # (G, 77, D)
        tokenized_prompts = self.tokenized_prompts  # (G, 77)，prompt BPE码，用于定位EOS。
        text_features = self.text_encoder(prompts, tokenized_prompts)  # (G, D)，从prompt_learner得到各类别prompt emb。

        # * 获取图像特征和中间量+注意力图。
        out = self.image_encoder(img.to(self.dtype), pad_info=pad_info)

        # * 计算前向到softmax logits。
        image_features = out.img_emb  # (N, D)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()  # (N, G) logit。

        out.logits = logits
        return out

    def forward(self, img: torch.Tensor, fg_cls_lb: torch.Tensor, pad_info: dict[str, ...]=None) -> Dict:
        # * 前向计算logits。
        out = self.get_logits(img, pad_info=pad_info)

        # * 计算softmax后的logits。
        logits = out.logits
        fg_num = fg_cls_lb.shape[1]
        match self.sm_fg_exist:
            case True:
                mask = torch.ones_like(logits, dtype=torch.bool)
                mask[:, :fg_num] = fg_cls_lb
                out.sm_logits = sm_logits = self.softmax(logits, mask=mask)
            case False:
                mask = None
                out.sm_logits = sm_logits = self.softmax(logits, mask=mask)
            case 'fg_only':
                mask = torch.zeros_like(logits, dtype=torch.bool)
                mask[:, :fg_num] = fg_cls_lb
                one_lb_mask = (fg_cls_lb.sum(dim=1) == 1)

                sm_logits = self.softmax(logits, mask=mask)
                sm_logits[one_lb_mask, :] = logits[one_lb_mask, :]
                out.sm_logits = sm_logits
            case _:
                raise ValueError(f"不支持的sm_fg_exist参数：{self.sm_fg_exist}")

        # * 拿出所有正样本的logits。计算正样本logits的数量以及所在样本编号。
        pos_logits = sm_logits[:, :fg_num][fg_cls_lb.to(torch.bool)]  # (pos_num,）取出所有正类logit。
        pos_num = pos_logits.shape[0]  # 正类logit的数目。
        pos_bt_idx, _ = torch.nonzero(fg_cls_lb, as_tuple=True)  # (pos_num,)的正类logit所在样本号和前景类别。

        # * 求每个正类logit关于激活值的导数。
        gcam_avt = out.gcam_avt  # (L, N, D)
        bt_grad_fg_logits = torch.eye(pos_num,
                                      dtype=pos_logits.dtype, device=pos_logits.device)  # (pos_num, pos_num)
        # torch._C._debug_only_display_vmap_fallback_warnings(True)  # batch grad的警告。
        grad_gcam_avt = grad(pos_logits, (gcam_avt,), grad_outputs=(bt_grad_fg_logits,),
                             create_graph=(self.mode == 'train'), is_grads_batched=True)[0]  # (pos_num, L, N, D)

        # * 找到每个正类logit的对应样本，求对应样本之patch激活值/梯度。
        # TODO 实验avt带梯度。
        pos_patch_avt = gcam_avt.detach()[1:, pos_bt_idx, :].permute(1, 0, 2)  # (L-1)PD -> (pos_num, L-1, D)，无梯度。
        pos_grad_patch_avt = grad_gcam_avt[torch.arange(pos_num), 1:, pos_bt_idx, :]  # (pos_num, L-1, D)

        # * 求每个正样本的CAM权重。
        pos_gcam_weight = torch.mean(pos_grad_patch_avt, dim=1, keepdim=True)  # (pos_num, 1, D)

        # * 求每个正样本的grad CAM。
        pos_gcam = pos_patch_avt * pos_gcam_weight  # (pos_num, L-1, D)，激活值各通道乘以权重。
        pos_gcam = torch.sum(pos_gcam, dim=2)  # (pos_num, L-1)，各通道平均。

        # * 将gcam变形到标准尺寸。
        gcam_h, gcam_w = out.emb_h, out.emb_w
        pos_gcam = pos_gcam.view(pos_num, gcam_h, gcam_w)

        out.pos_cam = pos_gcam
        return out

    @property
    def mode(self):
        return self._mode

    def set_mode(self, mode: 'str'):
        self._mode = mode
        match mode:
            case 'train':
                self.requires_grad_(False)
                self.eval()
                self.prompt_learner.requires_grad_(True)
                self.prompt_learner.train()
            case 'eval':
                self.requires_grad_(False)
                self.eval()
            case _:
                raise ValueError(f"不支持的{mode=}。")
        return self
