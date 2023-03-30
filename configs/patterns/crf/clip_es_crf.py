#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/29 14:18
@File    : clip_es_crf.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np

from alchemy_cat import RGB2BGR
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/patterns/crf/base.py')

cfg.crf.ini.iter_max = 10
cfg.crf.ini.pos_w = 3
cfg.crf.ini.pos_xy_std = 1
cfg.crf.ini.bi_w = 4
cfg.crf.ini.bi_xy_std = 67
cfg.crf.ini.bi_rgb_std = 3

@cfg.crf.ini.set_func()  # noqa
def img_preprocess(image: np.ndarray):
    image = RGB2BGR(image).astype(np.float32)
    image = image - (104.008, 116.669, 122.675)
    image = image.transpose(2, 0, 1)
    image = image.astype(np.uint8).transpose(1, 2, 0)  # 理论上会导致溢出orz，但试试看吧。
    return image
