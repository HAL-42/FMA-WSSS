#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/6/21 22:02
@File    : clip_es.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config()

cfg.model.fg_names = ['person with clothes,people,human', 'bicycle', 'car', 'motorbike', 'aeroplane',
                      'bus', 'train', 'truck', 'boat', 'traffic light',
                      'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird avian',
                      'cat', 'dog', 'horse', 'sheep', 'cow',
                      'elephant', 'bear', 'zebra', 'giraffe', 'backpack,bag',
                      'umbrella,parasol', 'handbag,purse', 'necktie', 'suitcase', 'frisbee',
                      'skis', 'sknowboard', 'sports ball', 'kite', 'baseball bat',
                      'glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                      'wine glass', 'cup', 'fork', 'knife', 'dessertspoon',
                      'bowl', 'banana', 'apple', 'sandwich', 'orange',
                      'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                      'cake', 'chair seat', 'sofa', 'pottedplant', 'bed',
                      'diningtable', 'toilet', 'tvmonitor screen', 'laptop', 'mouse',
                      'remote control', 'keyboard', 'cell phone', 'microwave', 'oven',
                      'toaster', 'sink', 'refrigerator', 'book', 'clock',
                      'vase', 'scissors', 'teddy bear', 'hairdrier,blowdrier', 'toothbrush',
                      ]

cfg.model.bg_names = ['ground', 'land', 'grass', 'tree', 'building', 'wall', 'sky', 'lake', 'water', 'river', 'sea',
                      'railway', 'railroad', 'helmet',
                      'cloud', 'house', 'mountain', 'ocean', 'road', 'rock', 'street', 'valley', 'bridge',
                      ]  # 相比VOC，去掉了keyboard和sign。
