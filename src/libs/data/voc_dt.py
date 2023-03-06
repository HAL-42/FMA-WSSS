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

from addict import Dict

from alchemy_cat.contrib.voc import VOCAug


class VOCAug2(VOCAug):
    """在标准VOCAug基础上:

    1. 返回cls_lb。
    2. 将图片改为RGB模式。
    3. 将输出打包为字典。
    """
    def __init__(self, root: str = "./contrib/datasets", year="2012", split: str = "train",
                 cls_labels_type: str='seg_cls_labels'):
        super().__init__(root, year, split)
        # * 参数检查与记录。
        assert cls_labels_type in ('seg_cls_labels', 'det_cls_labels', 'ignore_diff_cls_labels')
        self.cls_labels_type = cls_labels_type
        # * 读取图像级标签。
        with open(osp.join(self.root, 'third_party', f'{cls_labels_type}.pkl'), 'rb') as pkl_f:
            self.id2cls_labels = pickle.load(pkl_f)

    def get_item(self, index: int) -> Dict:
        img_id, img, lb = super().get_item(index)

        out = Dict()
        out.img_id, out.img, out.lb = img_id, img, lb
        out.cls_lb = self.id2cls_labels[img_id]

        return out
