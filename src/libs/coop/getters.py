#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/9 14:54
@File    : loader.py
@Software: PyCharm
@Desc    : 设计思路：模型组装、模型param获取、state_dict获取弱相关，故用getters中函数统一获取。
           模型要可以无痛无痛load保存的state_dict，故state_dict的键保留所有前缀。
           param_getter获取优化参数，结合模型，返回优化组。
"""
from typing import Callable

from collections import OrderedDict

from torch import nn

from libs.clip import load
from .custom_clips import cam_clips


def _get_coop_state(model: nn.Module) -> dict:
    new_state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        if 'prompt_learner.ctx' in k:
            new_state_dict[k] = v
    return new_state_dict


def _get_coop_sgd_named_param_groups(model: nn.Module, lr: float, weight_decay: float) -> dict[str, dict]:
    named_param_groups = {'new': {'params': [p for n, p in model.named_parameters() if 'prompt_learner' in n],
                                  'lr': lr,
                                  'weight_decay': weight_decay
                                  }}
    return named_param_groups


def grad_cam_clip(clip_name: str, fp32: bool,
                  classnames: list[str], ctx_cfg: dict,
                  adaptive_pos_emb: bool=False,
                  sm_fg_exist: bool=True) -> tuple[nn.Module, Callable, Callable]:
    clip_model, _ = load(clip_name, device='cpu', jit=False, cpu_float=fp32)
    gcam_clip = cam_clips.GradCAMCLIP(clip_model,
                                      classnames, ctx_cfg,
                                      adaptive_pos_emb=adaptive_pos_emb,
                                      sm_fg_exist=sm_fg_exist)
    return gcam_clip, _get_coop_state, _get_coop_sgd_named_param_groups
