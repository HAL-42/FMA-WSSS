#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/6/21 21:30
@File    : coco.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config, IL

from libs.data import COCO

cfg = config = Config('configs/patterns/coco_names/clip_es.py')

# * 设定数据集。
dt = cfg.dt.set_whole(True)
# ** 设定验证集。
dt.val.ini.cls_labels_type = 'seg_cls_labels'
dt.val.ini.split = 'train'
dt.val.ini.subsplit = ''
dt.val.dt = IL(lambda c:
               COCO(root='datasets', **c.dt.val.ini),
               priority=-1)

# * 设定保存于可视化。
cfg.solver.viz_step = 1000
