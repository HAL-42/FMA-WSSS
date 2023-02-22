#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/2/22 20:19
@File    : voc上色.py
@Software: PyCharm
@Desc    : 
"""
import os
import os.path as osp
import argparse
import multiprocessing as mp

from tqdm import tqdm
from cv2 import cv2

from alchemy_cat.acplot import RGB2BGR
from alchemy_cat.contrib.voc import label_map2color_map

import sys
sys.path = ['.', './src'] + sys.path  # noqa: E402


def 上色_label_file(label_file: str):
    label = cv2.imread(osp.join(args.source, label_file), cv2.IMREAD_GRAYSCALE)
    color_label = RGB2BGR(label_map2color_map(label))
    cv2.imwrite(osp.join(args.target, label_file), color_label)


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', type=str)
parser.add_argument('-t', '--target', type=str)
args = parser.parse_args()

os.makedirs(args.target, exist_ok=True)

with mp.Pool(int(mp.cpu_count() * 0.8)) as p:
    for _ in tqdm(p.imap_unordered(上色_label_file, os.listdir(args.source), chunksize=10),
                  dynamic_ncols=True, desc="处理", unit="张"):
        pass
