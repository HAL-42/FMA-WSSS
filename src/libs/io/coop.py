#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/9 21:17
@File    : coop.py
@Software: PyCharm
@Desc    : 
"""
from . import comm


def voc_inp_to_gcam_clip(inp):
    """CUDA字典值，并获取前景图像级标签。"""
    inp = comm.dict_to_cuda(inp)
    inp.fg_cls_lb = inp.ol_cls_lb[:, 1:] if ('ol_cls_lb' in inp) else inp.cls_lb[:, 1:]
    return inp


def gcam_clip_out_to_cls_loss(inp, out):
    """获取gcam_clip的输出，处理之，使其适配分类损失。"""
    fg_num = inp.fg_cls_lb.shape[1]
    out.fg_logits = out.logits[:, :fg_num]
    return out
