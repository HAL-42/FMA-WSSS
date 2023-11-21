#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/11/14 21:26
@File    : DECATHLON标签可视化.py
@Software: PyCharm
@Desc    : 
"""
import argparse
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="DECATHLON标签可视化")
parser.add_argument('--root', type=Path, required=True, help='DECATHLON数据集根目录')
parser.add_argument('--alpha', type=float, default=0.5, help='标签透明度')
parser.add_argument('--modality', type=str, default='ADC', choices=['ADC', 'T2'], help='模态')
args = parser.parse_args()

kColorMap: Final = np.array([[0, 0, 0], [0, 0, 255], [0, 255, 0]], dtype=np.uint8)

gt_dirs = list(args.root.glob('**/GT'))


def viz_gt(gt_dir: Path):
    (viz_dir := gt_dir / '..' / 'viz' / f'{args.modality}-GT').mkdir(exist_ok=True, parents=True)

    for lb_png in tqdm(list(gt_dir.glob('*.png')), dynamic_ncols=True):
        lb = cv2.imread(str(lb_png), cv2.IMREAD_GRAYSCALE)
        color_lb = kColorMap[lb]

        img = cv2.imread(str(gt_dir / '..' / args.modality / lb_png.name), cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        viz = cv2.addWeighted(img, 1 - args.alpha, color_lb, args.alpha, 0)

        cv2.imwrite(str(viz_dir / lb_png.name), viz)


for d in gt_dirs:
    print(f"Viz {d}...")
    viz_gt(d)
