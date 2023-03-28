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
import os
import pickle
from collections import OrderedDict
from functools import partial
from os import path as osp
from typing import Optional, Iterable, Callable, Union, Tuple, Any

import cv2
import numpy as np
from alchemy_cat.contrib.metrics import SegmentationMetric
from alchemy_cat.py_tools import OneOffTracker, Config
from frozendict import frozendict as fd
from tqdm import tqdm

__all__ = ['eval_cams', 'search_and_eval']


def eval_cams(class_num: int, class_names: Optional[Iterable[str]],
              cam_dir: str, preds_ignore_label: int,
              gt_dir: str, gts_ignore_label: int,
              cam2pred: Callable[[np.ndarray, Any, np.ndarray], np.ndarray],
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
        gt_dir: Dictionary where ground truths stored
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
        gt_file = osp.join(gt_dir, f'{img_id}.png')

        # Read files
        gt = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
        assert gt is not None
        gt: np.ndarray = gt_preprocess(gt)

        loaded = np.load(cam_file)
        cam, fg_cls = loaded['cam'].astype(np.float32), loaded['fg_cls'].astype(np.uint8)
        pred = cam2pred(cam, gt.shape, fg_cls)

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


def search_and_eval(dt, cam_dir: str, seed_cfg: Config, eval_dir: str):
    """搜索不同bg_method，获取最优性能。

    Args:
        dt: 数据集，提供类别数、类别名、忽略标签、标签目录等信息。
        cam_dir: cam文件所在目录。
        eval_dir: 保存评价结果的目录。
        seed_cfg: seed_cfg.cal(cam, gt.shape, fg_cls, seed_cfg.bg_methods[i], **seed_cfg.ini)将CAM转换为seed。

    Returns:
        None
    """
    # * 若已经有bg_method_metrics.pkl，则直接读取。
    if osp.isfile(bg_method_metrics_pkl := osp.join(eval_dir, 'bg_method_metrics.pkl')):
        with open(bg_method_metrics_pkl, 'rb') as f:
            bg_method_metrics = pickle.load(f)
    else:
        bg_method_metrics = {}

    # * 对各配置中的methods，计算其metric。
    for bg_method in seed_cfg.bg_methods:
        bg_method = fd(dict(bg_method))  # 将dict转换为frozendict，以便作为字典的key。
        if bg_method in bg_method_metrics:
            continue

        metric = eval_cams(class_num=dt.class_num,
                           class_names=dt.class_names,
                           cam_dir=cam_dir,
                           preds_ignore_label=255,
                           gt_dir=dt.label_dir,
                           gts_ignore_label=dt.ignore_label,
                           cam2pred=partial(seed_cfg.cal,
                                            bg_method=bg_method, **seed_cfg.ini),
                           result_dir=None,
                           importance=0,
                           eval_individually=False,
                           take_pred_ignore_as_a_cls=False)
        print(f'Current mIoU: {metric.mIoU:.4f} (bg_method={bg_method})')
        bg_method_metrics[bg_method] = metric

    # * 保存method_metrics.pkl。
    with open(bg_method_metrics_pkl, 'wb') as f:
        pickle.dump(bg_method_metrics, f)

    # * 遍历method_metrics字典，找到最好的metric。
    bg_methods, metrics = list(bg_method_metrics.keys()), list(bg_method_metrics.values())

    best_idx = np.argmax(mIoUs := [metric.mIoU for metric in metrics])
    best_metric = metrics[best_idx]
    best_metric.save_statistics(eval_dir, importance=0)

    # * 打印所有评价结果。
    for bg_method, mIoU in zip(bg_methods, mIoUs):
        print(f'mIoU: {mIoU:.4f} (bg_method={bg_method})')
    print(f'Best mIoU: {mIoUs[best_idx]:.4f} (bg_method={bg_methods[best_idx]})')
