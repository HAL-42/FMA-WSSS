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
from math import ceil

from alchemy_cat.py_tools import ADict
from cv2 import cv2
import numpy as np
from PIL import Image
from addict import Dict
from alchemy_cat.acplot import BGR2RGB, RGB2BGR
from alchemy_cat.data import Dataset, Subset

__all__ = ['COCO_NAMES', 'COCO_COLOR', 'COCO']


COCO_NAMES = ['background',
              'person', 'bicycle', 'car', 'motorcycle', 'airplane',
              'bus', 'train', 'truck', 'boat', 'traffic light',
              'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
              'cat', 'dog', 'horse', 'sheep', 'cow',
              'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
              'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
              'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
              'wine glass', 'cup', 'fork', 'knife', 'spoon',
              'bowl', 'banana', 'apple', 'sandwich', 'orange',
              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
              'cake', 'chair', 'couch', 'potted plant', 'bed',
              'dining table', 'toilet', 'tv', 'laptop', 'mouse',
              'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
              'toaster', 'sink', 'refrigerator', 'book', 'clock',
              'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
              ]

COCO_COLOR = np.array([(0, 0, 0),
                       (192, 0, 192), (64, 96, 0), (0, 0, 32), (64, 96, 64), (64, 96, 128),
                       (0, 0, 160), (0, 0, 224), (64, 0, 32), (64, 0, 96), (64, 0, 160),
                       (64, 0, 224), (192, 160, 0), (192, 160, 128), (192, 160, 192), (0, 128, 0),
                       (192, 192, 32), (0, 128, 64), (0, 128, 128), (192, 192, 160), (192, 96, 0),
                       (0, 128, 192), (64, 128, 0), (192, 128, 32), (64, 128, 64), (192, 128, 96),
                       (192, 96, 128), (64, 128, 128), (192, 128, 160), (192, 0, 32), (64, 128, 192),
                       (192, 128, 224), (0, 64, 0), (192, 0, 96), (0, 64, 64), (192, 0, 160),
                       (0, 64, 128), (192, 0, 224), (0, 64, 192), (0, 0, 0), (64, 64, 0),
                       (64, 64, 64), (0, 224, 0), (64, 64, 192), (0, 224, 128), (0, 224, 192),
                       (0, 192, 32), (0, 192, 96), (0, 192, 160), (0, 96, 0), (192, 192, 0),
                       (0, 192, 224), (64, 224, 0), (64, 192, 32), (64, 224, 64), (0, 96, 128),
                       (64, 192, 96), (64, 224, 128), (64, 192, 160), (0, 96, 192), (192, 192, 192),
                       (64, 160, 0), (0, 64, 32), (64, 160, 64), (0, 64, 96), (64, 160, 128),
                       (0, 64, 160), (64, 160, 192), (0, 64, 224), (64, 64, 32), (64, 64, 160),
                       (128, 0, 0), (192, 224, 0), (192, 224, 64), (128, 0, 128), (192, 224, 128),
                       (128, 0, 192), (0, 192, 0), (0, 192, 64), (0, 192, 128), (0, 192, 192)], dtype=np.uint8)

label2color = np.ones((256, 3), dtype=np.uint8) * 255
label2color[:81, :] = COCO_COLOR


class COCO(Dataset):
    """
    coco Segmentation base dataset
    """
    class_names = COCO_NAMES
    class_num = len(class_names)
    ignore_label = 255

    def __init__(self, root: str="./contrib/datasets", year="2014", split: str="train", subsplit: str="",
                 cls_labels_type: str='seg_cls_labels',
                 ps_mask_dir: str=None,
                 rgb_img: bool=True,
                 PIL_read: bool=True):
        # * 记录路径参数。
        self.root = root
        self.year = year
        assert split in ['train', 'val']
        self.split = split
        if subsplit:
            if split == 'train':
                assert subsplit in ['1250']
            if split == 'val':
                assert subsplit in ['1000', '5000']
        self.subsplit = subsplit

        # * 图像级标签选择。
        assert cls_labels_type in ('seg_cls_labels', 'det_cls_labels', 'ignore_diff_cls_labels')
        self.cls_labels_type = cls_labels_type

        # * 记录伪真值目录。
        self.ps_mask_dir = ps_mask_dir

        # * 是否返回RGB图像。
        self.rgb_img = rgb_img

        # * 是否用PIL读取。
        self.PIL_read = PIL_read

        # * 初始化属性。
        self.image_ids = []

        self.image_dir: str | None = None
        self.label_dir: str | None = None

        # * 读取元信息。
        self._set_files()

        # * 读取图像级标签。
        with open(osp.join(self.root, 'third_party', f'{cls_labels_type}.pkl'), 'rb') as pkl_f:
            self.id2cls_labels = pickle.load(pkl_f)

    @classmethod
    def subset(cls,
               split_idx_num: tuple[int, int],
               root: str="./contrib/datasets", year="2014", split: str="train",
               cls_labels_type: str='seg_cls_labels',
               ps_mask_dir: str=None,
               rgb_img: bool=True,
               PIL_read: bool=True):
        dt = cls(root, year, split, cls_labels_type, ps_mask_dir, rgb_img, PIL_read)

        split_idx, split_num = split_idx_num
        assert split_idx < split_num

        step = ceil(len(dt) / split_num)
        indexes = list(range(split_idx * step, min((split_idx + 1) * step, len(dt))))

        sub_dt = Subset(dt, indexes)
        return sub_dt

    def _set_files(self):
        self.root = osp.join(self.root, f'coco{self.year}')
        self.image_dir = osp.join(self.root, 'images', f'{self.split}{self.year}')
        self.label_dir = osp.join(self.root, 'annotations', f'{self.split}{self.year}')

        set_id_file = osp.join(self.root, 'imagesLists',
                               f'{self.split}' + (f'_{self.subsplit}' if self.subsplit else '')  + '.txt')
        with open(set_id_file, 'r') as f:
            self.image_ids = [id_.rstrip() for id_ in f.readlines()]

    def __len__(self):
        return len(self.image_ids)

    def get_by_img_id(self, img_id: str):
        try:
            index = self.image_ids.index(img_id)
        except ValueError:
            raise RuntimeError(f"Can't find img_id {img_id} in dataset's image_ids list")
        return self[index]

    def get_item(self, index: int) -> Dict:
        # * Load image
        img_id = self.image_ids[index]
        img_path = osp.join(self.image_dir, f'{img_id}.jpg')
        if self.PIL_read:
            img = np.asarray(Image.open(img_path), dtype=np.uint8)
            if not self.rgb_img:
                img = RGB2BGR(img).copy()
        else:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if self.rgb_img:
                img = BGR2RGB(img).copy()

        # * Load Label
        lb_path = osp.join(self.label_dir, f'{img_id}.png')
        lb = np.asarray(Image.open(lb_path), dtype=np.uint8)

        # * 构造输出。
        out = ADict()
        out.img_id, out.img = img_id, img
        out.cls_lb = self.id2cls_labels[img_id]

        if self.ps_mask_dir is not None:
            out.lb = np.asarray(Image.open(osp.join(self.ps_mask_dir, f'{img_id}.png')), dtype=np.uint8)
        else:
            out.lb = lb

        return out

    @staticmethod
    def label_map2color_map(label_map: np.ndarray) -> np.ndarray:
        return label2color[label_map]