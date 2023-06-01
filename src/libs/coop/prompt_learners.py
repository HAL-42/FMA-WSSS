#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/7 16:24
@File    : prompt_learner.py
@Software: PyCharm
@Desc    : 拷贝自coop的trainer。
"""
from typing import Any

import numpy as np
import torch
from alchemy_cat.py_tools import set_torch_rand_seed
from alchemy_cat.torch_tools import RNGCacher
from torch import nn

from libs import clip
from libs.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


class CoOpLearner(nn.Module):

    def __init__(self, classnames: list[str], clip_model,
                 n_ctx: int,
                 ctx_init: str,
                 csc: bool | list | np.ndarray,
                 cls_token_pos: str,
                 ctx_std: float=0.0125):
        super().__init__()
        n_cls = len(classnames)
        # n_ctx = cfg.TRAINER.COOP.N_CTX
        # ctx_init = cfg.TRAINER.COOP.CTX_INIT
        # dtype = clip_model.dtype
        dtype = torch.float32  # NOTE 自有参数全部为float32，因为amp不支持fp16的梯度更新。
        ctx_dim = clip_model.ln_final.weight.shape[0]
        # clip_imsize = clip_model.visual.input_resolution
        # cfg_imsize = cfg.INPUT.SIZE[0]
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            match csc:
                case True:
                    print("Initializing class-specific contexts")
                    ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)  # (G, M, D)
                case False:
                    print("Initializing a generic context")
                    ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)  # (M, D)
                case (list() | np.ndarray()):
                    print("Initializing a mixin context")
                    assert len(csc) == len(classnames)
                    ctx_vectors = torch.empty(np.amax(csc) + 1, n_ctx, ctx_dim, dtype=dtype)  # (G, M, D)
                case _:
                    raise ValueError(f"Unknown csc type: {csc}")

            nn.init.normal_(ctx_vectors, std=ctx_std)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized, 类型与clip_model一致

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]  # name为空，则长度为0。
        prompts = [prompt_prefix + " " + name + "." for name in classnames]  # '.'和' .'的编号是一样，name为空不影响。

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (G, 77)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # (G, 1, D)的SOS, 类型与clip_model一致
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # (G, 60, D)的类名+EOS, 类型与clip_model一致

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.csc = csc
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cls_token_pos
        self.ctx_std = ctx_std

    def initialize(self, seed: Any):
        with RNGCacher():
            set_torch_rand_seed(seed)
            nn.init.normal_(self.ctx, std=self.ctx_std)

    def forward(self):
        ctx = self.ctx  # (M, D) 或 (G, M, D)

        match self.csc:
            case False:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # (G, M, D)
            case (list() | np.ndarray()):
                ctx = ctx[self.csc, :, :]  # (G, M, D)
            case _:
                pass

        prefix = self.token_prefix  # (G, 1, D)的SOS
        suffix = self.token_suffix  # (G, 60, D)的CLS+类名+EOS

        if self.class_token_position == "end":  # 合并SOS、可学上下文、CLS、EOS。
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]  # (1, 1, dim)
                class_i = suffix[i: i + 1, :name_len, :]  # (1, 'class name', dim)
                suffix_i = suffix[i: i + 1, name_len:, :]  # (1, '. <end> 0 0 ...', dim)
                ctx_i = ctx[i: i + 1, :, :]  # (1, n_ctx, dim)
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts
