#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/13 16:37
@File    : seed_thresh.py
@Software: PyCharm
@Desc    : 
"""
from typing import Callable

import torch
import torch.nn.functional as F
import numpy as np

from .score import cam2score, cam2score_cuda


def _parse_dsize(dsize):
    if isinstance(dsize, np.ndarray) and dsize.ndim == 3:  # dsize是HWC的图片。
        return dsize.shape[:2], dsize
    else:
        return dsize, None


def seed_argmax(cam: np.ndarray, dsize, fg_cls: np.ndarray, bg_method: dict, resize_first: bool,
                crf: Callable[[np.ndarray, np.ndarray], np.ndarray] | None=None) -> (np.ndarray, np.ndarray):
    dsize, img = _parse_dsize(dsize)

    score = cam2score(cam, dsize=dsize, resize_first=resize_first)

    cls_lb = np.pad(fg_cls + 1, (1, 0), mode='constant', constant_values=0)

    match bg_method:
        case {'method': 'thresh', 'thresh': thresh}:
            bg_score = np.full_like(score[0], thresh)[None, ...]
        case {'method': 'pow', 'pow': p}:
            bg_score = np.power(1 - np.max(score, axis=0, keepdims=True), p)
        case _:
            raise ValueError(f'Unknown bg_method: {bg_method}')

    score = np.concatenate((bg_score, score), axis=0)

    if crf is not None:
        score = crf(img, score)

    seed = np.argmax(score, axis=0)
    seed = cls_lb[seed].astype(np.uint8)

    return seed, score


# TODO 支持batch输入。
def seed_argmax_cuda(cam: np.ndarray | torch.Tensor, dsize, fg_cls: np.ndarray | torch.Tensor,
                     bg_method: dict, resize_first: bool,
                     crf: Callable[[np.ndarray, np.ndarray], np.ndarray] | None=None) -> (np.ndarray, np.ndarray):
    dsize, img = _parse_dsize(dsize)

    cam = torch.asarray(cam, device='cuda', requires_grad=False)
    fg_cls = torch.asarray(fg_cls, device='cuda', requires_grad=False)

    score = cam2score_cuda(cam, dsize=dsize, resize_first=resize_first)

    cls_lb = F.pad(fg_cls + 1, (1, 0), mode='constant', value=0)

    match bg_method:
        case {'method': 'thresh', 'thresh': thresh}:
            bg_score = torch.full_like(score[0], thresh)[None, ...]
        case {'method': 'pow', 'pow': p}:
            bg_score = torch.pow(1 - torch.amax(score, dim=0, keepdim=True), p)
        case _:
            raise ValueError(f'Unknown bg_method: {bg_method}')

    score = torch.cat((bg_score, score), dim=0)

    if crf is not None:
        score = torch.from_numpy(crf(img, score.cpu().numpy())).to(device='cuda')

    seed = torch.argmax(score, dim=0)
    seed = cls_lb[seed].to(torch.uint8)

    return seed.cpu().numpy(), score.cpu().numpy()
