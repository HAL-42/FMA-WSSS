#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/28 20:32
@File    : utils.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np
from cv2 import cv2

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


def scoremap2bbox(scoremap: np.ndarray, threshold: float, multi_contour_eval: bool=True) -> (np.ndarray, int):
    """将scoremap转为BBox。

    Args:
        scoremap: (H, W)的scoremap，其值应当在0-1之间。
        threshold: 二值化阈值。
        multi_contour_eval: 是否返回scoremap中的所有轮廓，为False只返回最大的轮廓。

    Returns:
        BBox的左上角和右下角坐标，以及轮廓数目。
    """
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)  # 255整数化，(H, W, 1)。
    _, thr_gray_heatmap = cv2.threshold(  # 按阈值二值化为0/255。
        src=scoremap_image,
        thresh=int(threshold * np.max(scoremap_image)),
        maxval=255,
        type=cv2.THRESH_BINARY)
    contours = cv2.findContours(  # 二值图寻找轮廓，RETE_TREE返回所有轮廓及其层次（无论内外层），method表轮廓上的线段只记端点。
        image=thr_gray_heatmap,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]  # 得到tuple(np.array[端点数目, 1, 2])的轮廓端点。

    if len(contours) == 0:
        return np.asarray([[0, 0, 0, 0]]), 1

    if not multi_contour_eval:
        contours = [max(contours, key=cv2.contourArea)]

    estimated_boxes = []
    for contour in contours:  # 将所有轮廓转为BBox——尽管大部分BBox只有1x1大小😂。
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0, x1, y1 = x, y, x + w, y + h
        x1 = min(x1, width - 1)  # 防止越界。
        y1 = min(y1, height - 1)
        estimated_boxes.append([x0, y0, x1, y1])

    return np.asarray(estimated_boxes), len(contours)  # np.array[轮廓数, 4]
