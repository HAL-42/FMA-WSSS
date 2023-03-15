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
import multiprocessing as mp

import cv2
import numpy as np

from alchemy_cat.data.plugins import identical
from alchemy_cat.contrib.metrics import SegmentationMetric
from alchemy_cat.py_tools import OneOffTracker

__all__ = ['eval_cams']


def _get_pred_gt(suffix_cam_dir_gts_dir_cam2pred_gt_preprocess):
    suffix, cam_dir, gts_dir, cam2pred, gt_preprocess = suffix_cam_dir_gts_dir_cam2pred_gt_preprocess

    # Set files
    img_id = osp.splitext(suffix)[0]
    cam_file = osp.join(cam_dir, suffix)
    gt_file = osp.join(gts_dir, f'{img_id}.png')

    # Read files
    loaded = np.load(cam_file)
    cam, fg_cls = loaded['cam'], loaded['fg_cls']
    pred = cam2pred(cam, fg_cls)

    gt = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
    assert gt is not None
    gt: np.ndarray = gt_preprocess(gt)
    assert pred.shape == gt.shape

    return img_id, pred, gt


def eval_cams(class_num: int, class_names: Optional[Iterable[str]],
              cam_dir: str, preds_ignore_label: int,
              gts_dir: str, gts_ignore_label: int,
              cam2pred: Callable[[np.ndarray, np.ndarray], np.ndarray],
              result_dir: Optional[str]=None,
              gt_preprocess: Callable[[np.ndarray], np.ndarray]=identical,
              importance: int=2,
              eval_individually: bool=True, take_pred_ignore_as_a_cls: bool=False,
              metric_cls: type=SegmentationMetric,
              pool_size: int=4) \
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
        pool_size: Pool size for multiprocessing. (Default: 4)

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

    with mp.Pool(pool_size) as pool:
        imap_iter = pool.imap_unordered(_get_pred_gt,
                                        [(suffix, cam_dir, gts_dir, cam2pred, gt_preprocess)
                                         for suffix in cam_file_suffixes],
                                        chunksize=4)

        for img_id, pred, gt in tqdm(imap_iter,
                                     total=len(cam_file_suffixes),
                                     desc='eval progress', unit='sample', dynamic_ncols=True):
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
