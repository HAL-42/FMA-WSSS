#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/20 15:50
@File    : cls_m_IoU,pra.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config()

cfg.metric_names = ['mIoU', 'macro_avg_precision', 'macro_avg_recall', 'accuracy', 'cls_IoU']
