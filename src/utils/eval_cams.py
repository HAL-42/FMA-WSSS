#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/13 16:20
@File    : eval_cams.py
@Software: PyCharm
@Desc    : 
"""
from typing import Optional, Iterable, Callable, Union, Tuple
import os
from os import path as osp
import pickle
from collections import OrderedDict
from tqdm import tqdm

import cv2
import numpy as np

from alchemy_cat.contrib.metrics import SegmentationMetric
from alchemy_cat.py_tools import OneOffTracker

__all__ = ['eval_cams']


def eval_cams(class_num: int, class_names: Optional[Iterable[str]],
              cam_dir: str, preds_ignore_label: int,
              gts_dir: str, gts_ignore_label: int,
              cam2pred: Callable[[np.ndarray, np.ndarray], np.ndarray],
              result_dir: Optional[str]=None,
              gt_preprocess: Callable[[np.ndarray], np.ndarray]=lambda x: x,
              importance: int=2,
              eval_individually: bool=True, take_pred_ignore_as_a_cls: bool=False,
              metric_cls: type=SegmentationMetric) \
        -> Union[Tuple[SegmentationMetric, OrderedDict], SegmentationMetric]:

    """Evaluate predictions of semantic segmentation

    Args:
        class_num: Num of classes
        class_names: Name of classes
        cam_dir: Dictionary where cam stored
        preds_ignore_label: Ignore label for predictions
        gts_dir: Dictionary where ground truths stored
        gts_ignore_label: Ignore label for ground truths
        result_dir: If not None, eval result will be saved to result_dir. (Default: None)
        cam2pred: Function transfer cam file to prediction.
        gt_preprocess: Preprocess function for ground truth read in
        importance: Segmentation Metric's importance filter. (Default: 2)
        eval_individually: If True, evaluate each sample. (Default: True)
        take_pred_ignore_as_a_cls: If True, the ignored label in preds will be seemed as a class. (Default: False)
        metric_cls: Use metric_cls(class_num, class_names) to eval preds. (Default: SegmentationMetric)

    Returns:
        Segmentation Metric and metric result for each sample (If eval_individually is True)
    """
    assert preds_ignore_label >= class_num
    assert gts_ignore_label >= class_num

    if take_pred_ignore_as_a_cls:
        class_num += 1
        class_names += ['pred_ignore']

    metric = metric_cls(class_num, class_names)
    if eval_individually:
        sample_metrics = OrderedDict()

    print("\n================================== Eval ==================================")
    cam_file_suffixes = os.listdir(cam_dir)
    for cam_file_suffix in tqdm(cam_file_suffixes,
                                total=len(cam_file_suffixes), miniters=10,
                                desc='eval progress', unit='sample', dynamic_ncols=True):
        # Set files
        img_id = osp.splitext(cam_file_suffix)[0]
        cam_file = osp.join(cam_dir, cam_file_suffix)
        gt_file = osp.join(gts_dir, f'{img_id}.png')

        # Read files
        loaded = np.load(cam_file)
        cam, fg_cls = loaded['cam'].astype(np.float32), loaded['fg_cls'].astype(np.float32)
        pred = cam2pred(cam, fg_cls)

        gt = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
        assert gt is not None
        gt: np.ndarray = gt_preprocess(gt)
        assert pred.shape == gt.shape

        if take_pred_ignore_as_a_cls:
            pred[pred == preds_ignore_label] = class_num - 1
            if gts_ignore_label == class_num - 1:
                gt[gt == gts_ignore_label] = class_num

        # Evaluate
        metric.update(pred, gt)
        if eval_individually:
            with OneOffTracker(lambda: metric_cls(class_num, class_names)) as individual_metric:
                individual_metric.update(pred, gt)
            sample_metrics[img_id] = individual_metric.statistics(importance)

    # Saving
    if result_dir is not None:
        metric.save_metric(result_dir, importance, dpi=400)
        if eval_individually:
            with open(osp.join(result_dir, 'sample_statistics.pkl'), 'wb') as f:
                pickle.dump(sample_metrics, f)

    print("\n================================ Eval End ================================")

    if eval_individually:
        return metric, sample_metrics
    else:
        return metric
