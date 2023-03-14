#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/13 11:15
@File    : clip_es.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config()


cfg.model.fg_names = ['aeroplane', 'bicycle', 'bird avian', 'boat', 'bottle',
                      'bus', 'car', 'cat', 'chair seat', 'cow',
                      'diningtable', 'dog', 'horse', 'motorbike', 'person with clothes,people,human',
                      'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor screen',
                      ]
cfg.model.bg_names = ['ground', 'land', 'grass', 'tree', 'building',
                      'wall', 'sky', 'lake', 'water', 'river',
                      'sea', 'railway', 'railroad', 'keyboard', 'helmet',
                      'cloud', 'house', 'mountain', 'ocean', 'road',
                      'rock', 'street', 'valley', 'bridge', 'sign',
                      ]
