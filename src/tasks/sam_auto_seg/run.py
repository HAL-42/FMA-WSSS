#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/17 22:39
@File    : run.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import os
import os.path as osp
import pickle
import sys
import traceback

import matplotlib.pyplot as plt
from alchemy_cat.py_tools import rprint
from alchemy_cat.torch_tools import init_env
from tqdm import tqdm

from segment_anything import sam_model_registry

sys.path = ['.', './src'] + sys.path  # noqa: E402

from libs.viz.viz_anns import show_imgs_anns

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config')
parser.add_argument('-b', '--points_per_batch', default=256, type=int)
parser.add_argument('-w', '--rock_sand_water_chunk_size', default=50, type=int)
parser.add_argument('-m', '--output_mode', default='uncompressed_rle', type=str)
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
os.makedirs(ann_save_dir := osp.join(cfg.rslt_dir, 'anns'), exist_ok=True)

levels_viz_dirs = []
for levels in cfg.viz.level_combs:
    os.makedirs(levels_viz_dir := osp.join(cfg.rslt_dir, 'viz', 'masks',
                                           f'level={",".join(str(l) for l in levels)}'),
                exist_ok=True)
    levels_viz_dirs.append(levels_viz_dir)

# * 初始化数据集。
dt = cfg.dt.cls(**cfg.dt.ini)
print(dt, end="\n\n")

# * 初始化SAM基础模型。
sam = sam_model_registry[cfg.sam.model_type](checkpoint=cfg.sam.checkpoint)
sam = sam.to(device=device)
print(sam, end="\n\n")

# * 初始化SAM自动分割模型。
mask_generator = cfg.mask_gen.cls(model=sam,
                                  points_per_batch=args.points_per_batch,
                                  rock_sand_water_chunk_size=args.rock_sand_water_chunk_size,
                                  output_mode=args.output_mode,
                                  **cfg.mask_gen.ini)
print(mask_generator, end="\n\n")

# * 遍历所有图片，生成掩码并可视化。
fig: plt.Figure = plt.figure(dpi=300)

for idx, inp in tqdm(enumerate(dt), total=len(dt), dynamic_ncols=True, desc='推理', unit='张', miniters=10):
    img_id, img = inp.img_id, inp.img

    if osp.isfile(anns_file := osp.join(ann_save_dir, f'{img_id}.pkl')):
        try:
            with open(anns_file, 'rb') as f:
                _ = pickle.load(f)
        except Exception as e:
            rprint(f"[重算] {img_id} 存在但无法加载，重新计算。")
        else:
            print(f"[重算] {img_id} 已经存在且可以被正确加载。")
            continue

    try:
        img_anns = mask_generator.generate(img)
    except Exception as e:
        rprint(f"对{img_id}运行SAM自动分割失败！")
        print(traceback.format_exc())
        continue

    # * 保存掩码。
    with open(anns_file, 'wb') as f:
        pickle.dump(img_anns, f)

    # * 可视化。
    if cfg.viz.enable and (idx % cfg.viz.step == 0):
        for levels, levels_viz_dir in zip(cfg.viz.level_combs, levels_viz_dirs, strict=True):
            fig.clf()
            show_imgs_anns(fig, [img], [img_id], [img_anns], levels=levels, alpha=cfg.viz.alpha)

            if args.show_viz:
                fig.show()

            fig.savefig(osp.join(levels_viz_dir, f'{img_id}.png'), bbox_inches='tight')
