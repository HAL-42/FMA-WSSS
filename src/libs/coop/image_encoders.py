#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/7 16:36
@File    : image_encoders.py
@Software: PyCharm
@Desc    : 
"""
from dataclasses import dataclass

from addict import Dict

import torch
from torch import nn
import torch.nn.functional as F

from libs.clip.model import CLIP, ResidualAttentionBlock


def scale_pos_emb(pos_emb: torch.Tensor, emb_h: int, emb_w: int,
                  pad_pos: list[tuple[int, int, int, int]]=None) -> torch.Tensor:
    """将位置嵌入缩放到与嵌入相同尺寸。

    Args:
        pos_emb: (L, D)的位置嵌入。
        emb_h: 分块嵌入的高。
        emb_w: 分块嵌入的高。
        pad_pos: 若输入图像被填充，原图位置的坐标。

    Returns:
        (emb_h * emb_w + 1, D)的缩放后位置嵌入。
    """
    # * 将位置嵌入分解为类别嵌入和分块嵌入。
    cls_pos_emb = pos_emb[:1, :]  # (1, D)
    patch_pos_emb = pos_emb[1:, :]  # (L-1, D)

    # * 将分块位置嵌入转为(1, D, sqrt(L), sqrt(L))形式。
    L, D = patch_pos_emb.shape
    S = int(L ** .5)
    assert S * S == L
    # ** 若没有pad，则尺寸相同，不需要scale。
    if (pad_pos is None) and ((emb_h == S) and (emb_w == S)):
        return pos_emb
    # ** 若有pad，判断条件太复杂，干脆总是scale。
    patch_pos_emb = patch_pos_emb.permute(1, 0)  # (D, L-1)
    patch_pos_emb = patch_pos_emb.view(1, D, S, S)  # (1, D, S, S)

    # * 上采样位置嵌入。
    if pad_pos is None:
        scaled_patch_pos_emb = F.interpolate(patch_pos_emb, size=(emb_h, emb_w), mode='bilinear', align_corners=False)
        scaled_patch_pos_emb = scaled_patch_pos_emb.view(D, -1)  # (D, hw)
        scaled_patch_pos_emb = scaled_patch_pos_emb.permute(1, 0)  # (hw, D)
        scaled_pos_emb = torch.cat([cls_pos_emb, scaled_patch_pos_emb], 0)  # (hw + 1, D)
    else:
        n = len(pad_pos)
        scaled_patch_pos_emb = torch.zeros(n, D, emb_h, emb_w, device=patch_pos_emb.device, dtype=patch_pos_emb.dtype)
        for idx, (i, j, img_h, img_w) in enumerate(pad_pos):
            patch_pos_emb_on_img = F.interpolate(patch_pos_emb, size=(img_h, img_w),
                                                 mode='bilinear', align_corners=False)[0]  # (D, img_h, img_w)
            scaled_patch_pos_emb[idx, :, i:i + img_h, j:j + img_w] = patch_pos_emb_on_img
        scaled_patch_pos_emb = scaled_patch_pos_emb.view(n, D, -1)  # (n, D, hw)
        scaled_patch_pos_emb = scaled_patch_pos_emb.permute(0, 2, 1)  # (n, hw, D)
        scaled_pos_emb = torch.cat([cls_pos_emb.expand(n, 1, D), scaled_patch_pos_emb], 1)  # (n, hw + 1, D)

    # scaled_pos_emb = nn.parameter.Parameter(scaled_pos_emb.half())  # 理论上不会改变类型。缩放后emb应当是中间量而非参数。

    # * 返回。
    assert scaled_pos_emb.dtype == pos_emb.dtype
    return scaled_pos_emb


@dataclass
class _EncoderCache(object):
    n: int | None = None
    d: int | None = None
    ori_h: int | None = None
    ori_w: int | None = None
    emb_h: int | None = None
    emb_w: int | None = None
    L: int | None = None
    key_padding_mask: torch.Tensor | None = None
    pad_pos: list[tuple[int, int, int, int]] | None = None

    def clear(self):
        self.n, self.d, self.ori_h, self.ori_w = None, None, None, None
        self.emb_h, self.emb_w = None, None
        self.L = None
        self.key_padding_mask = None
        self.pad_pos = None


class GetLN1(nn.Module):
    """
    返回：
    1. 各层注意力图。
    2. 最后一层LN1的输出。
    3. 类别牌嵌入。
    4. 各patch嵌入。
    5. 全图patch平均嵌入。
    """

    def __init__(self, clip_model: CLIP, adaptive_pos_emb: bool=False):
        super().__init__()
        self.visual = clip_model.visual
        self.dtype = clip_model.dtype
        self.adaptive_pos_emb = adaptive_pos_emb
        self._cache = _EncoderCache()

    def mount_mask(self, x: torch.Tensor, pad_info: dict[str, ...]=None):
        # * 预先计算尺寸，并记录到cache。
        n, d, ori_h, ori_w = x.shape
        patch_size = self.visual.patch_size
        emb_h, emb_w = ori_h // patch_size, ori_w // patch_size
        L = emb_h * emb_w + 1

        self._cache.n, self._cache.d, self._cache.ori_h, self._cache.ori_w = n, d, ori_h, ori_w
        self._cache.emb_h, self._cache.emb_w = emb_h, emb_w
        self._cache.L = L

        # * 生成mask。
        if pad_info is not None:
            mask = pad_info['padding_mask']  # NHW
            assert mask.shape == (n, ori_h, ori_w)
            # mask = F.unfold(mask[:, None, :, :], kernel_size=patch_size, stride=patch_size)  # (N, p^2, L-1)
            mask = mask.view(n, ori_h // patch_size, patch_size, ori_w // patch_size, patch_size)
            mask = mask.permute(0, 1, 3, 2, 4).contiguous().view(n, -1, patch_size * patch_size)  # (N, L-1, p^2)
            mask = torch.all(mask, dim=2)  # (N, L-1)，若patch全为pad，则为True。
            mask = torch.cat([torch.zeros(n, 1, device=mask.device, dtype=mask.dtype), mask], dim=1)  # (N, L)
            self._cache.key_padding_mask = mask
        else:
            mask = None

        # * 生成有效位置。
        if self.adaptive_pos_emb and mask is not None:
            img_mask = (~mask[:, 1:]).view(n, emb_h, emb_w)  # (N, H, W)

            pad_pos = []
            for m in img_mask:  # (H, W)
                h_idxes, w_idxes = torch.nonzero(m, as_tuple=True)
                i, ii, j, jj = h_idxes.min(), h_idxes.max(), w_idxes.min(), w_idxes.max()
                img_h, img_w = ii - i + 1, jj - j + 1
                pad_pos.append((i, j, img_h, img_w))

            self._cache.pad_pos = pad_pos  # [(i, j, h, w), ...], N个

        # * 挂载（清除）mask。
        for m in self.modules():
            if isinstance(m, ResidualAttentionBlock):
                m.key_padding_mask = mask

    def run_stem(self, x: torch.Tensor) -> torch.Tensor:
        """完成patch-emb、位置先验。"""
        visual = self.visual
        # * 分块嵌入。
        x = visual.conv1(x)  # shape = [*, width, grid, grid] NDHW
        # * 检验patch嵌入图尺寸。
        emb_h, emb_w = x.shape[2:]
        assert (self._cache.emb_h, self._cache.emb_w) == (emb_h, emb_w)
        # ** 拉平。
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2] ND(L-1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width] N(L-1)D
        # * 拼接上class token。
        x = torch.cat(
            [visual.class_embedding.to(x.dtype) +
             torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width] NLD
        # * 加上位置嵌入。
        pos_emb = (scale_pos_emb(visual.positional_embedding, emb_h, emb_w, self._cache.pad_pos)
                   if self.adaptive_pos_emb
                   else visual.positional_embedding)
        x = x + pos_emb.to(x.dtype)
        # * LN并返回。
        x = visual.ln_pre(x)
        return x

    def run_trans(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """通过transformer，获得每一层的attention map以及最后一层att前ln后的输出。"""
        transformer = self.visual.transformer

        # * 通过前N-1层trans，积累注意力图。
        att_weights = []
        for i in range(transformer.layers - 1):
            x, att_weight = transformer.resblocks[i](x, need_weights=True)
            att_weights.append(att_weight)
        # * 逐步通过最后一层，获取ln1后特征。
        last_block = transformer.resblocks[-1]
        gcam_avt: torch.Tensor = last_block.ln_1(x)  # LGD, ln1的输出。
        gcam_avt.requires_grad_()
        att_out, att_weight = last_block.attention(gcam_avt, need_weights=True)
        att_weights.append(att_weight)  # 保存最后一层注意力图。
        x = x + att_out
        x = x + last_block.mlp(last_block.ln_2(x))

        return x, att_weights, gcam_avt

    def run_post(self, x: torch.Tensor):
        """完成最后的LN和投影。"""
        visual = self.visual

        x = visual.ln_post(x)  # NLD

        cls_emb = x[:, 0, :]  # ND，类别牌的嵌入。
        patch_emb = x[:, 1:, :]  # N(L-1)D，所有patch的嵌入。
        if (mask := self._cache.key_padding_mask) is None:
            img_emb = torch.mean(patch_emb, dim=1)  # ND，图片的总嵌入。
        else:
            img_mask = ~mask[:, 1:]  # N(L-1)
            img_emb = (torch.sum(patch_emb * img_mask[:, :, None], dim=1) /  # ND
                       img_mask.sum(dim=1, keepdim=True, dtype=patch_emb.dtype))  # ND，图片的总嵌入。

        if visual.proj is not None:
            cls_emb = cls_emb @ visual.proj
            patch_emb = patch_emb @ visual.proj
            img_emb = img_emb @ visual.proj

        return cls_emb, patch_emb, img_emb

    def forward(self, x: torch.Tensor, pad_info: dict[str, ...]=None):
        # * 清空cache。
        self._cache.clear()

        # * 挂载mask。
        self.mount_mask(x, pad_info)

        # * stem前向，图片转嵌入。
        x = self.run_stem(x.to(self.dtype))

        # * 嵌入做attention。
        # ** 转为LGD适配trans输入。
        x = x.permute(1, 0, 2)  # NLD -> LND
        # ** trans前向。
        x, att_weights, gcam_avt = self.run_trans(x)
        # ** 转为GLD，适配后处理.
        x = x.permute(1, 0, 2)  # LND -> NLD

        # * 投影。
        cls_emb, patch_emb, img_emb = self.run_post(x)  # ND, N(L-1)D, ND
        # * 将ln1输出和分块嵌入变形到适合CAM计算。
        # emb_h, emb_w = self._cache.emb_h, self._cache.emb_w
        #
        # gcam_avt = gcam_avt.permute(1, 2, 0)  # (L-1)ND -> ND(L-1)
        # gcam_avt = gcam_avt.view(gcam_avt.shape[0], gcam_avt.shape[1], emb_h, emb_w)  # ND(L-1) -> NDHW
        #
        # patch_emb = patch_emb.permute(0, 2, 1)  # N(L-1)D -> ND(L-1)
        # patch_emb = patch_emb.view(patch_emb.shape[0], patch_emb.shape[1], emb_h, emb_w)  # ND(L-1) -> NDHW

        # * 构造输出。
        out = Dict()
        out.emb_h, out.emb_w = self._cache.emb_h, self._cache.emb_w
        out.att_weights, out.gcam_avt = att_weights, gcam_avt
        out.cls_emb, out.patch_emb, out.img_emb = cls_emb, patch_emb, img_emb
        return out
