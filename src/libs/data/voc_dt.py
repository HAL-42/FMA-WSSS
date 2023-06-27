#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/5 14:13
@File    : voc_dt.py
@Software: PyCharm
@Desc    : 
"""
import os.path as osp
import pickle

import numpy as np
from PIL import Image
from addict import Dict
from alchemy_cat.acplot import BGR2RGB
from alchemy_cat.contrib.voc import VOCAug, label_map2color_map
from alchemy_cat.data import Subset
from math import ceil


class VOCAug2(VOCAug):
    """在标准VOCAug基础上:

    1. 返回cls_lb。
    2. 支持将lb替换为伪标签。
    3. 将输出打包为字典。
    4. 支持返回RGB img。
    """
    def __init__(self, root: str = "./contrib/datasets", year="2012", split: str = "train",
                 cls_labels_type: str='seg_cls_labels',
                 ps_mask_dir: str=None,
                 rgb_img: bool=False):
        super().__init__(root, year, split, PIL_read=True)
        # * 参数检查与记录。
        assert cls_labels_type in ('seg_cls_labels', 'det_cls_labels', 'ignore_diff_cls_labels')
        self.cls_labels_type = cls_labels_type
        # * 读取图像级标签。
        with open(osp.join(self.root, 'third_party', f'{cls_labels_type}.pkl'), 'rb') as pkl_f:
            self.id2cls_labels = pickle.load(pkl_f)
        # * 记录伪真值目录。
        self.ps_mask_dir = ps_mask_dir
        # * 是否返回RGB图像。
        self.rgb_img = rgb_img

    def get_item(self, index: int) -> Dict:
        img_id, img, lb = super().get_item(index)

        if self.rgb_img:
            img = BGR2RGB(img).copy()

        out = Dict()
        out.img_id, out.img, out.lb = img_id, img, lb

        if self.split != 'test':
            out.cls_lb = self.id2cls_labels[img_id]
        else:
            out.cls_lb = np.zeros((self.class_num,), dtype=np.uint8)

        if self.ps_mask_dir is not None:
            out.lb = np.asarray(Image.open(osp.join(self.ps_mask_dir, f'{img_id}.png')), dtype=np.uint8)

        return out

    @classmethod
    def subset(cls,
               split_idx_num: tuple[int, int],
               root: str = "./contrib/datasets", year="2012", split: str = "train",
               cls_labels_type: str='seg_cls_labels',
               ps_mask_dir: str=None,
               rgb_img: bool=False):
        dt = cls(root, year, split, cls_labels_type, ps_mask_dir, rgb_img)

        split_idx, split_num = split_idx_num
        assert split_idx < split_num

        step = ceil(len(dt) / split_num)
        indexes = list(range(split_idx * step, min((split_idx + 1) * step, len(dt))))

        sub_dt = Subset(dt, indexes)
        return sub_dt

    @staticmethod
    def label_map2color_map(label_map: np.ndarray) -> np.ndarray:
        return label_map2color_map(label_map)
