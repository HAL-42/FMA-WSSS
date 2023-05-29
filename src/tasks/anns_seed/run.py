#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/23 21:54
@File    : run.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import os
import os.path as osp
import pickle
import sys

import numpy as np
import torch
from PIL import Image
from alchemy_cat.contrib.evaluation.semantics_segmentation import eval_preds
from alchemy_cat.contrib.voc import label_map2color_map
from alchemy_cat.data.plugins import arr2PIL
from alchemy_cat.torch_tools import init_env
from matplotlib import pyplot as plt
from tqdm import tqdm

sys.path = ['.', './src'] + sys.path  # noqa: E402

from libs.sam import SamAuto, SamAnns
from libs.viz.viz_anns import show_anns
from utils.resize import resize_cam_cuda
from utils.cache_dir import CacheDir

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
os.makedirs(seed_dir := CacheDir(osp.join(cfg.rslt_dir, 'seed'),
                                 '/tmp/anns_seed/seed',
                                 exist='delete'),
            exist_ok=True)
if cfg.viz.enable:
    os.makedirs(color_seed_dir := CacheDir(osp.join(cfg.rslt_dir, 'viz', 'color_seed_dir'),
                                           '/tmp/anns_seed/color_seed',
                                           exist='delete'),
                exist_ok=True)
    os.makedirs(img_cam_seed_dir := osp.join(cfg.rslt_dir, 'viz', 'img_cam_seed_dir'), exist_ok=True)
os.makedirs(eval_dir := osp.join(cfg.rslt_dir, 'eval'), exist_ok=True)

# * 初始化数据集。
dt = cfg.dt.cls(**cfg.dt.ini)
print(dt, end="\n\n")
fg_names = dt.class_names[1:]

# * 遍历所有图片，生成优化后掩码并可视化。
fig: plt.Figure = plt.figure(dpi=300)

for idx, inp in tqdm(enumerate(dt), total=len(dt), dynamic_ncols=True, desc='生成', unit='张', miniters=10):
    img_id, img, lb = inp.img_id, inp.img, inp.lb
    ori_h, ori_w = img.shape[:2]

    # * 读取CAM和前景类别。
    loaded = np.load(osp.join(cfg.cam.dir, f'{img_id}.npz'))
    cam = torch.as_tensor(loaded['cam'], dtype=torch.float32, device=device)  # PHW
    fg_cls = torch.as_tensor(loaded['fg_cls'], dtype=torch.uint8, device=device)  # P

    # ** CAM插值到原图大小。
    cam = resize_cam_cuda(cam, (ori_h, ori_w))

    # * 读取SAM标注，并计算种子点。
    if osp.isfile(anns_file := osp.join(cfg.sam_anns.dir, f'{img_id}.pkl')):
        with open(anns_file, 'rb') as f:
            anns = SamAnns(pickle.load(f))

        # * 提前准备好二进制掩码segmentation。
        for ann in anns:
            SamAuto.decode_segmentation(ann, replace=True)

        # * stack好数据。
        anns.stack_data(masks=True, areas=True, device=device)

        seed = cfg.seed.cal(anns, cam, fg_cls).cpu().numpy()
    else:
        print(f'{img_id}没有对应标注, 使用替补种子点。')
        anns = None
        seed = np.asarray(Image.open(osp.join(cfg.seed.dir, f'{img_id}.png')), dtype=np.uint8)
        seed[seed == 255] = 0  # seed中Ignore部分一律为背景。否则eval结果不精确。

    # * 保存优化后种子点。
    arr2PIL(seed).save(osp.join(seed_dir, f'{img_id}.png'))

    # * 可视化。
    if cfg.viz.enable:
        color_seed = label_map2color_map(seed)
        arr2PIL(color_seed, order='RGB').save(osp.join(color_seed_dir, f'{img_id}.png'))

    if cfg.viz.enable and (idx % cfg.viz.step == 0):
        fig.clf()

        row_num = 4
        col_num = cam.shape[0]

        for col, (c, cls) in enumerate(zip(cam.cpu().numpy(), fg_cls.cpu().numpy())):

            # * 图片 + 真值。
            ax: plt.axes = fig.add_subplot(row_num, col_num, 0 * col_num + col + 1)
            ax.imshow(img)
            ax.imshow(label_map2color_map(lb), alpha=0.5)
            ax.set_title(img_id, fontsize='smaller')
            ax.axis("off")

            # * 图片 + anns。
            ax: plt.axes = fig.add_subplot(row_num, col_num, 1 * col_num + col + 1)
            ax.imshow(img)
            if anns is not None:
                show_anns(anns, ax=ax, alpha=0.5)
            ax.set_title(f"{len(anns)} masks", fontsize='smaller')
            ax.axis('off')

            # * 图片 + cam。
            ax: plt.axes = fig.add_subplot(row_num, col_num, 2 * col_num + col + 1)
            ax.imshow(img)
            ax.imshow(c, cmap=plt.get_cmap('jet'), alpha=0.5)
            ax.set_title(fg_names[cls], fontsize='smaller', color='red', fontweight='bold')
            ax.axis("off")

            # * seed。
            ax: plt.axes = fig.add_subplot(row_num, col_num, 3 * col_num + col + 1)
            ax.imshow(color_seed)
            ax.set_title('seed', fontsize='smaller')
            ax.axis("off")

        if args.show_viz:
            fig.show()

        fig.savefig(osp.join(img_cam_seed_dir, f'{img_id}.png'), bbox_inches='tight')

seed_dir.flush()
if cfg.viz.enable:
    color_seed_dir.flush()

# * 评价种子点性能。
metric = eval_preds(class_num=dt.class_num,
                    class_names=dt.class_names,
                    preds_dir=str(seed_dir),
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

seed_dir.save(no_flush=True)
if cfg.viz.enable:
    color_seed_dir.save(no_flush=True)
