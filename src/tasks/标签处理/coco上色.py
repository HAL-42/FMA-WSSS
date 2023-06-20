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
import argparse
import multiprocessing as mp
import os
import os.path as osp
import sys

import numpy as np
from PIL import Image
from alchemy_cat.contrib.evaluation.semantics_segmentation import eval_preds
from alchemy_cat.data.plugins import arr2PIL
from tqdm import tqdm

sys.path = ['.', './src'] + sys.path  # noqa: E402

from libs.data import COCO


def 上色_label_file(label_file: str):
    if not label_file.endswith('.png'):
        return

    label = np.array(Image.open(osp.join(args.source, label_file)), dtype=np.uint8)
    color_label = dt.label_map2color_map(label)
    arr2PIL(color_label).save(osp.join(args.target, label_file))


def ignore2bg(lb: np.ndarray):
    lb[lb == 255] = 0
    return lb


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--eval', type=int, default=0)
parser.add_argument('-s', '--source', type=str)
parser.add_argument('-t', '--target', type=str)
parser.add_argument('--ts', type=int, default=0)
args = parser.parse_args()

os.makedirs(args.target, exist_ok=True)

dt = COCO('datasets')

with mp.Pool(int(mp.cpu_count() * 0.2)) as p:
    for _ in tqdm(p.imap_unordered(上色_label_file, os.listdir(args.source), chunksize=10),
                  dynamic_ncols=True, desc="处理", unit="张"):
        pass

if args.eval:
    metric = eval_preds(class_num=dt.class_num,
                        class_names=dt.class_names,
                        preds_dir=str(args.source),
                        preds_ignore_label=dt.ignore_label,
                        gts_dir=dt.label_dir,
                        gts_ignore_label=dt.ignore_label,
                        pred_preprocess=ignore2bg,
                        result_dir=None,
                        importance=0,
                        eval_individually=False,
                        take_pred_ignore_as_a_cls=False)

    metric.print_statistics(0)
