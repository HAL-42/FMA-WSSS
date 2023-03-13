#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/9 21:17
@File    : comm.py
@Software: PyCharm
@Desc    : 
"""
import torch


def dict_to_cuda(inp: dict):
    for k, v in inp.items():
        if torch.is_tensor(v):
            inp[k] = v.to('cuda', non_blocking=True)
        if isinstance(v, dict):
            dict_to_cuda(v)
    return inp
