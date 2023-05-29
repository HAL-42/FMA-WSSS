#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/18 11:04
@File    : viz_anns.py
@Software: PyCharm
@Desc    : 
"""
from typing import Iterable, Any

import matplotlib.pyplot as plt
import numpy as np
from alchemy_cat.acplot import square

from libs.sam.custom_sam.sam_auto import SamAuto

__all__ = ['show_anns', 'show_imgs_anns']


def show_anns(anns, ax: plt.Axes=None, alpha: float=0.5):
    if len(anns) == 0:
        return

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    ax = plt.gca() if ax is None else ax
    ax.set_autoscale_on(False)

    img = np.ones((*sorted_anns[0]['img_hw'], 4))
    img[:, :, 3] = 0

    for ann in sorted_anns:
        m = SamAuto.decode_segmentation(ann, replace=False)
        color_seed = int(ann['area']) + 999 * int(sum(ann['bbox'])) + 999999 * int(sum(ann['point_coords'][0]))
        color_seed %= 2 ** 32
        color_mask = np.concatenate([np.random.RandomState(color_seed).random(3), [alpha]])
        img[m] = color_mask

    ax.imshow(img)


def show_imgs_anns(fig: plt.Figure,
                   imgs: Iterable[np.ndarray], img_ids: Iterable[str],
                   anns: Iterable[list[dict[str, Any]]],
                   levels: tuple[int, ...]=(0, 1, 2), alpha: float=0.5):
    img_ids = list(img_ids)

    row_num, col_num = square(len(list(imgs)))

    for i, (img, ann) in enumerate(zip(imgs, anns, strict=True)):
        ann = [a for a in ann if a['level'] in levels]  # 滤出指定level的anns。

        ax: plt.Axes = fig.add_subplot(row_num, col_num, i + 1)
        ax.imshow(img)
        show_anns(ann, ax=ax, alpha=alpha)
        ax.set_title(f"{img_ids[i]}: {len(ann)} masks", fontsize=5)
        ax.axis('off')

    fig.tight_layout()
