#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/4/6 21:19
@File    : base.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/clip_cam/coco_base.py',  # COCO基础训练配置。
                      'configs/clip_cam/_patches/cls_only.py',  # CAM上损失权重为0，无在线标签。
                      'configs/clip_cam/_patches/cl_loss.py')  # 使用MultiLabelCLLoss作为损失函数。
