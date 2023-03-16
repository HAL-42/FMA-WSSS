#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/7 16:32
@File    : text_encoder.py
@Software: PyCharm
@Desc    : 
"""
import torch
from torch import nn

from libs.clip.model import CLIP


class EncEmb(nn.Module):
    def __init__(self, clip_model: CLIP):  # 提取text编码器部分的子模块：位置编码、Transformer、LN、投影头。
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor):  # 直接输入emb好的prompts，而不是编码。
        x = prompts.type(self.dtype) + self.positional_embedding.type(self.dtype)  # (G, 77, D)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)  # LN只处理D，所以L、N顺序无所谓。

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
