#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/27 22:18
@File    : run.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import multiprocessing as mp
import os
import os.path as osp
import subprocess
import sys
import uuid
from glob import glob

import matplotlib
import numpy as np
import torch
from alchemy_cat.acplot import BGR2RGB, col_all
from alchemy_cat.contrib.tasks.wsss.viz import viz_cam
from alchemy_cat.py_tools import get_local_time_str
from alchemy_cat.torch_tools import init_env
from matplotlib import pyplot as plt
from natsort import natsorted, ns
from tqdm import tqdm

sys.path = ['.', './src'] + sys.path  # noqa: E402

from libs.seeding.score import cam2score
from utils.cache_dir import CacheDir
from utils.eval_cams import search_and_eval
from utils.resize import resize_cam

mp.set_start_method('spawn', force=True)

if __name__ == '__main__':
    # * 读取命令行参数。
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    parser.add_argument("-d", '--is_debug', default=0, type=int)
    parser.add_argument('-s', '--show_viz', default=0, type=int)
    parser.add_argument('-e', '--eval_only', default=0, type=int)
    parser.add_argument('-P', '--pool_size', default=0, type=int)
    parser.add_argument('--cache_ori_cam', default=1, type=int)
    args = parser.parse_args()

    # * 初始化环境。
    device, cfg = init_env(is_cuda=True,
                           is_benchmark=False,
                           is_train=False,  # 需要求grad cam。
                           config_path=args.config,
                           experiments_root="experiment",
                           rand_seed=0,
                           cv2_num_threads=0,
                           verbosity=True,
                           log_stdout=True,
                           reproducibility=False,
                           is_debug=bool(args.is_debug))
    if not args.show_viz:
        matplotlib.use('Agg')
    print(f"{matplotlib.get_backend()=}")

    # * 配置路径。
    os.makedirs(cam_affed_cache_dir := osp.join('/tmp',
                                                'aff_cam',
                                                f'{uuid.uuid4().hex}@{get_local_time_str(for_file_name=True)}'),
                exist_ok=False)
    if cfg.solver.save_cam:
        cam_affed_save_dir = osp.join(cfg.rslt_dir, 'cam_affed')
    if cfg.solver.viz_cam:
        os.makedirs(cam_affed_viz_dir := osp.join(cfg.rslt_dir, 'viz', 'cam_affed'), exist_ok=True)
    if cfg.solver.viz_score:
        os.makedirs(score_viz_dir := osp.join(cfg.rslt_dir, 'viz', 'score'), exist_ok=True)

    # * 如果只需要eval，此时即可eval并退出。
    if args.eval_only:
        assert cfg.eval.enabled
        subprocess.run(['cp', '-a', osp.join(cam_affed_save_dir, '.'), cam_affed_cache_dir])
        search_and_eval(cfg.dt.val.dt, cam_affed_cache_dir, cfg.eval.seed, cfg.rslt_dir, args.pool_size)
        subprocess.run(['rm', '-r', cam_affed_cache_dir])
        exit(0)

    # * 将原始CAM文件拷贝到/tmp下。
    ori_cam_cache_dir = CacheDir(cfg.aff.ori_cam_dir, '/tmp/aff_cam/ori_cam', exist='cache',
                                 enabled=bool(args.cache_ori_cam))

    # * 数据集。
    val_dt = cfg.dt.val.dt
    print(val_dt, end="\n\n")
    fg_names = val_dt.class_names[1:]

    # * 准备可视化图像。
    fig = plt.figure(dpi=600)

    # * 逐个遍历CAM文件，优化、保存并可视化。
    for idx, cam_file in tqdm(enumerate(cam_files := natsorted(glob(osp.join(ori_cam_cache_dir, '*.npz')),
                                                               alg=ns.PATH)),
                              total=len(cam_files), dynamic_ncols=True, desc='优化', unit='批次', miniters=10):
        img_id = osp.splitext(osp.basename(cam_file))[0]

        # * 读取CAM。
        loaded = np.load(cam_file)
        ori_cam = torch.from_numpy(loaded['cam'].astype(np.float32)).to(device)  # PHW
        fg_cls = loaded['fg_cls'].astype(np.uint8)
        fg_logit = loaded['fg_logit'].astype(np.float32)
        all_logit = loaded['all_logit'].astype(np.float32)
        att = torch.from_numpy(loaded['att'].astype(np.float32)).to(device)  # DHL

        # * 优化CAM。
        cam_affed = cfg.aff.cal(att, ori_cam).cpu().numpy()

        # * 保存cam_affed。
        np.savez(osp.join(cam_affed_cache_dir, f'{img_id}.npz'),
                 cam=cam_affed, fg_cls=loaded['fg_cls'], fg_logit=fg_logit, all_logit=all_logit)

        # * 如果需要可视化，根据img_id获取原始图像。
        if (cfg.solver.viz_cam or cfg.solver.viz_score) and (idx % cfg.solver.viz_step == 0):
            ori_inp = val_dt.get_by_img_id(img_id)
            ori_img, ori_lb = BGR2RGB(ori_inp.img), ori_inp.lb
            ori_h, ori_w = ori_img.shape[:2]

        # * 可视化CAM。
        if cfg.solver.viz_cam and (idx % cfg.solver.viz_step == 0):
            fig.clf()

            pos_names = ['dummy']
            for cls, c in zip(fg_cls, cam_affed, strict=True):
                pos_names.append(f'{fg_names[cls]} {c.min():.1e} {c.max():.1e}')

            resized_cam = resize_cam(cam_affed, (ori_h, ori_w))

            viz_cam(fig=fig,
                    img_id=img_id, img=ori_img, label=ori_lb,
                    cls_in_label=np.ones(len(pos_names), dtype=np.uint8),
                    cam=resized_cam,
                    cls_names=pos_names,
                    get_row_col=col_all)

            if args.show_viz:
                fig.show()

            # ** 保存CAM可视化结果。
            fig.savefig(osp.join(cam_affed_viz_dir, f'{img_id}.png'), bbox_inches='tight')

        # * 可视化score。
        if cfg.solver.viz_score and (idx % cfg.solver.viz_step == 0):
            fig.clf()

            score = cam2score(cam_affed, (ori_h, ori_w), cfg.solver.viz_score.resize_first)

            pos_names = ['dummy']
            for cls, logit in zip(fg_cls, fg_logit, strict=True):
                pos_names.append(f'{fg_names[cls]} {logit:.1f}')

            viz_cam(fig=fig,
                    img_id=img_id, img=ori_img, label=ori_lb,
                    cls_in_label=np.ones(len(pos_names), dtype=np.uint8),
                    cam=score,
                    cls_names=pos_names,
                    get_row_col=col_all)

            if args.show_viz:
                fig.show()

            # ** 保存Score可视化结果。
            fig.savefig(osp.join(score_viz_dir, f'{img_id}.png'), bbox_inches='tight')

    # * 删除暂存的ori_cam。
    ori_cam_cache_dir.terminate()

    # * 如果需要保存CAM，则将暂存的cam_affed拷贝到保存目录。
    if cfg.solver.save_cam:
        if osp.isdir(cam_affed_save_dir):
            subprocess.run(['rm', '-r', cam_affed_save_dir])
        subprocess.run(['cp', '-a', cam_affed_cache_dir, cam_affed_save_dir])

    # * 如果只需要eval，此时即可eval并退出。
    if cfg.eval.enabled:
        torch.cuda.empty_cache()
        search_and_eval(cfg.dt.val.dt, cam_affed_cache_dir, cfg.eval.seed, cfg.rslt_dir, args.pool_size)

    # * 删除cam_affed暂存目录。
    subprocess.run(['rm', '-r', cam_affed_cache_dir])
