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
from itertools import product

import numpy as np
import torch
from alchemy_cat.torch_tools import init_env
from matplotlib import pyplot as plt

sys.path = ['.', './src'] + sys.path  # noqa: E402

from libs.sam import SamAuto, SamAnns
from utils.resize import resize_cam_cuda

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config')
parser.add_argument('-d', '--is_debug', default=0, type=int)
parser.add_argument('-s', '--show_viz', default=0, type=int)
args = parser.parse_args()

IMG_ID = '2007_003205'

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

# * 建立结果文件夹。
os.makedirs(plot_dir := osp.join(cfg.rslt_dir, 'sp_scores'), exist_ok=True)

# * 初始化数据集。
dt = cfg.dt.cls(**cfg.dt.ini)
print(dt, end="\n\n")
fg_names = dt.class_names[1:]

# * 初始化图片。
fig: plt.Figure = plt.figure(dpi=300)

for norm_first, bg_method in product(cfg.seed.norm_firsts, cfg.seed.bg_methods):
    # * 遍历所有图片，生成优化后掩码并可视化。
    inp = dt.get_by_img_id(IMG_ID)
    img_id, img, lb = inp.img_id, inp.img, inp.lb
    ori_h, ori_w = img.shape[:2]

    # * 读取CAM和前景类别。
    if cfg.cam.loader:
        cam, fg_cls = cfg.cam.loader.cal(cfg.cam.dir, img_id)
        cam = torch.as_tensor(cam, dtype=torch.float32, device=device)  # PHW
        fg_cls = torch.as_tensor(fg_cls, dtype=torch.uint8, device=device)  # P
    else:
        loaded = np.load(osp.join(cfg.cam.dir, f'{img_id}.npz'))
        cam = torch.as_tensor(loaded['cam'], dtype=torch.float32, device=device)  # PHW
        fg_cls = torch.as_tensor(loaded['fg_cls'], dtype=torch.uint8, device=device)  # P

    # ** CAM插值到原图大小。
    cam = resize_cam_cuda(cam, (ori_h, ori_w))

    # * 读取SAM标注，并计算种子点。
    with open(osp.join(cfg.sam_anns.dir, f'{img_id}.pkl'), 'rb') as f:
        anns = SamAnns(pickle.load(f))

    # * 提前准备好二进制掩码segmentation。
    for ann in anns:
        SamAuto.decode_segmentation(ann, replace=True)

    # * stack好数据。
    anns.stack_data(masks=True, areas=True, device=device)

    seed, seeded_anns = cfg.seed.cal(anns, cam, fg_cls,
                                     norm_first=norm_first, bg_method=bg_method, ret_seeded_anns=True)

    # * 画出sp得分。
    sp_scores = torch.zeros((cam.shape[0] + 1, ori_h, ori_w), dtype=cam.dtype, device=cam.device)

    for ann, m in zip(seeded_anns, seeded_anns.data.masks, strict=True):
        sp_scores[:, m] = ann['score'][:, None]

    fig.clf()

    names = ['background'] + [fg_names[c] for c in fg_cls.cpu().numpy()]

    for col, (score, name) in enumerate(zip(sp_scores.cpu().numpy(), names, strict=True)):
        # * 图片 + cam。
        ax: plt.axes = fig.add_subplot(cam.shape[0] + 1, 1, col + 1)
        # NOTE 由于色阶很少，CMAP应当是单色的，只改变亮度。排除了jet、hot等。
        # NOTE 单色的CMAP里，红色、黄色容易混淆——究竟是白色最大还是红色最大？
        # NOTE 使用蓝色（绿色丑），且最大值为白色。
        ax.imshow(1 - score, cmap=plt.get_cmap('Blues'), alpha=0.5, vmin=0, vmax=1)
        ax.set_title(name, fontsize='smaller', color='red', fontweight='bold')
        ax.axis("off")

    fig.show()
    plt.show()

    fig.savefig(osp.join(plot_dir, f'{IMG_ID}.png'), bbox_inches='tight')
