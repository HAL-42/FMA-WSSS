#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/18 21:03
@File    : run.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import os
import os.path as osp
import pickle
import sys
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import torch
from alchemy_cat.contrib.evaluation.semantics_segmentation import eval_preds
from alchemy_cat.contrib.voc import label_map2color_map
from alchemy_cat.data.plugins import arr2PIL
from alchemy_cat.torch_tools import init_env
from tqdm import tqdm

sys.path = ['.', './src'] + sys.path  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config')
parser.add_argument('-d', '--is_debug', default=0, type=int)
parser.add_argument('-s', '--show_viz', default=0, type=int)
args = parser.parse_args()

device, cfg = init_env(is_cuda=True,
                       is_benchmark=False,
                       is_train=False,
                       config_path=args.config,
                       experiments_root="experiment",
                       rand_seed=True,
                       cv2_num_threads=-1,
                       verbosity=True,
                       log_stdout=True,
                       reproducibility=False,
                       is_debug=bool(args.is_debug))

# * 初始化保存目录。
os.makedirs(refined_seed_save_dir := osp.join(cfg.rslt_dir, 'refined_seed'), exist_ok=True)
if cfg.viz.enable:
    os.makedirs(color_refined_seed_dir := osp.join(cfg.rslt_dir, 'viz', 'color_refined_seed'), exist_ok=True)
    os.makedirs(img_seed_refined_dir := osp.join(cfg.rslt_dir, 'viz', 'img_seed_refined'), exist_ok=True)
os.makedirs(eval_dir := osp.join(cfg.rslt_dir, 'eval'), exist_ok=True)

# * 初始化数据集。
dt = cfg.dt.cls(**cfg.dt.ini)
print(dt, end="\n\n")

# * 初始化投票器。
voter = cfg.voter.cls(**cfg.voter.ini)

# * 遍历所有图片，生成优化后掩码并可视化。
fig: plt.Figure = plt.figure(dpi=300)

for idx, inp in tqdm(enumerate(dt), total=len(dt), dynamic_ncols=True, desc='投票', unit='张', miniters=10):
    img_id, img, lb = inp.img_id, inp.img, inp.lb

    seed = np.asarray(Image.open(osp.join(cfg.seed.dir, f'{img_id}.png')), dtype=np.uint8)

    if osp.isfile(anns_file := osp.join(cfg.sam_anns.dir, f'{img_id}.pkl')):
        with open(anns_file, 'rb') as f:
            anns = pickle.load(f)

        refined_seed = voter.vote(seed=torch.as_tensor(seed, device=device), anns=anns).cpu().numpy()
    else:
        print(f'No annotation for {img_id}, use seed directly.')
        refined_seed = seed

    refined_seed[refined_seed == 255] = 0  # seed中Ignore部分一律为背景。否则eval结果不精确。

    # * 保存优化后种子点。
    arr2PIL(refined_seed).save(osp.join(refined_seed_save_dir, f'{img_id}.png'))

    # * 可视化。
    if cfg.viz.enable:
        color_refined_seed = label_map2color_map(refined_seed)
        arr2PIL(color_refined_seed, order='RGB').save(osp.join(color_refined_seed_dir, f'{img_id}.png'))

    if cfg.viz.enable and (idx % cfg.viz.step == 0):

        fig.clf()

        ax: plt.axes = fig.add_subplot(1, 3, 1)
        ax.imshow(img)
        ax.imshow(label_map2color_map(lb), alpha=0.5)
        ax.set_title(img_id, fontsize='smaller')
        ax.axis("off")

        ax: plt.axes = fig.add_subplot(1, 3, 2)
        ax.imshow(label_map2color_map(seed))
        ax.set_title('seed', fontsize='smaller')
        ax.axis("off")

        ax: plt.axes = fig.add_subplot(1, 3, 3)
        ax.imshow(color_refined_seed)
        ax.set_title('refined_seed', fontsize='smaller')
        ax.axis("off")

        if args.show_viz:
            fig.show()

        fig.savefig(osp.join(img_seed_refined_dir, f'{img_id}.png'), bbox_inches='tight')

# * 评价种子点性能。
metric = eval_preds(class_num=dt.class_num,
                    class_names=dt.class_names,
                    preds_dir=refined_seed_save_dir,
                    preds_ignore_label=dt.ignore_label,
                    gts_dir=dt.label_dir,
                    gts_ignore_label=dt.ignore_label,
                    result_dir=None,
                    importance=0,
                    eval_individually=False,
                    take_pred_ignore_as_a_cls=False)
print(metric)
print(f'mIoU: {metric.mIoU:.4f}')
metric.save_metric(eval_dir, importance=0, figsize=(24, 24))
