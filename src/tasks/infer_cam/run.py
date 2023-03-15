#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/10 21:23
@File    : run.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import os
import os.path as osp
import pickle
import sys
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from alchemy_cat.acplot import BGR2RGB, col_all
from alchemy_cat.contrib.tasks.wsss.viz import viz_cam
from alchemy_cat.torch_tools import init_env, update_model_state_dict
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path = ['.', './src'] + sys.path  # noqa: E402

from utils.eval_cams import eval_cams
from utils.norm import min_max_norm


def search_and_eval():
    dt = cfg.dt.val.dt

    # * 若已经有bg_method_metrics.pkl，则直接读取。
    if osp.isfile(bg_method_metrics_pkl := osp.join(eval_dir, 'bg_method_metrics.pkl')):
        with open(bg_method_metrics_pkl, 'rb') as f:
            bg_method_metrics = pickle.load(f)
    else:
        bg_method_metrics = {}

    # * 对各配置中的methods，计算其metric。
    for bg_method in cfg.eval.seed.bg_methods:
        if bg_method in bg_method_metrics:
            continue

        metric = eval_cams(class_num=dt.class_num,
                           class_names=dt.class_names,
                           cam_dir=cam_save_dir,
                           preds_ignore_label=255,
                           gts_dir='datasets/VOC2012/SegmentationClassAug',
                           gts_ignore_label=dt.ignore_label,
                           cam2pred=partial(cfg.eval.seed.cal, bg_method=bg_method),
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

    for bg_method, mIoU in zip(bg_methods, mIoUs):
        print(f'mIoU: {mIoU:.4f} (bg_method={bg_method})')
    print(f'Best mIoU: {mIoUs[best_idx]:.4f} (bg_method={bg_methods[best_idx]})')


# * 读取命令行参数。
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config')
parser.add_argument('-f', '--prefetch_factor', default=2, type=int)
parser.add_argument('-p', '--pin_memory', default=0, type=int)
parser.add_argument("-b", '--benchmark', default=0, type=int)
parser.add_argument("-d", '--is_debug', default=0, type=int)
parser.add_argument('-s', '--show_viz', default=0, type=int)
parser.add_argument('-e', '--eval_only', default=0, type=int)
args = parser.parse_args()

# * 初始化环境。
device, cfg = init_env(is_cuda=True,
                       is_benchmark=bool(args.benchmark),
                       is_train=True,  # 需要求grad cam。
                       config_path=args.config,
                       experiments_root="experiment",
                       rand_seed=0,
                       cv2_num_threads=0,
                       verbosity=True,
                       log_stdout=True,
                       reproducibility=False,
                       is_debug=bool(args.is_debug))

# * 配置路径。
cam_save_dir = osp.join(cfg.rslt_dir, 'cam')
if cfg.solver.save_cam:
    os.makedirs(cam_save_dir, exist_ok=True)
if cfg.solver.viz_cam:
    os.makedirs(cam_viz_dir := osp.join(cfg.rslt_dir, 'viz', 'cam'), exist_ok=True)
if cfg.solver.viz_score:
    os.makedirs(score_viz_dir := osp.join(cfg.rslt_dir, 'viz', 'score'), exist_ok=True)
if cfg.eval.enabled:
    os.makedirs(eval_dir := osp.join(cfg.rslt_dir, 'eval'), exist_ok=True)

# * 如果只需要eval，此时即可eval并退出。
if args.eval_only:
    assert cfg.eval.enabled
    search_and_eval()
    exit(0)

# * 数据集。
val_dt = cfg.dt.val.dt
print(val_dt, end="\n\n")

# * 训练数据增强器。
val_auger = cfg.auger.val.cls(val_dt, **cfg.auger.val.ini)
print(val_auger, end="\n\n")

# * 数据加载器。
val_loader = DataLoader(val_auger,
                        batch_size=cfg.loader.val.batch_size,
                        num_workers=cfg.loader.val.num_workers,
                        pin_memory=bool(args.pin_memory),
                        shuffle=False,
                        drop_last=False,
                        generator=torch.Generator().manual_seed(0),
                        prefetch_factor=args.prefetch_factor
                        )
print(val_loader, end="\n\n")
epoch_val_loader = iter(val_loader)

# * 分类模型。
model, _, _ = cfg.model.cls(**cfg.model.ini)
cal_model = cfg.model.val_cal

if resume_file := cfg.model.resume_file:
    update_model_state_dict(model, torch.load(resume_file, map_location='cpu'), verbosity=3)
print(model, end="\n\n")

model.set_mode('eval')
model = model.to(device)

# * 准备可视化图像。
fig = plt.figure(dpi=600)

idx = 0  # 用于计数推理了多少张图。

for inp in tqdm(val_loader, dynamic_ncols=True, desc='推理', unit='批次', miniters=10):
    # * 若只eval，跳过。
    if args.eval_only:
        break

    # * 获取新一个批次数据。
    inp = cfg.io.update_inp(inp)

    # * 前向。
    out = cal_model(model, inp)
    out = cfg.io.update_out(inp, out)

    # * 获取out中正类CAM pos_cam，转到CPU上，随后按照他们的batch_idx分组。
    pos_batch_idx = np.nonzero(fg_cls_lb := inp.fg_cls_lb.cpu().numpy())[0]
    pos_cam = out.pos_cam.cpu()  # PHW

    sample_cam = [pos_cam[pos_batch_idx == idx, :, :] for idx in range(val_loader.batch_size)]  # [样本数xHxW]
    sample_fg_cls = [np.nonzero(fg_cls_lb[i, :])[0] for i in range(val_loader.batch_size)]
    fg_logits = out.fg_logits.detach().cpu().numpy()
    sample_fg_logit = [fg_logits[i, fg_cls_lb[i, :].astype(bool)] for i in range(val_loader.batch_size)]

    # * 遍历每张图的id和CAM，保存到文件并可视化之。
    for img_id, cam, fg_cls, fg_logit in zip(inp.img_id, sample_cam, sample_fg_cls, sample_fg_logit, strict=True):
        # * 根据img_id获取原始图像。
        ori_inp = val_dt.get_by_img_id(img_id)
        ori_img, ori_lb = BGR2RGB(ori_inp.img), ori_inp.lb
        ori_h, ori_w = ori_img.shape[:2]

        # * 将CAM转到原始图像的尺寸上。
        cam = F.interpolate(cam.unsqueeze(0), size=(ori_h, ori_w), mode='bilinear', align_corners=False).squeeze(0)
        cam = cam.numpy().astype(np.float16)  # [样本数xHxW]

        # * 保存CAM。
        if cfg.solver.save_cam:
            np.savez(osp.join(cam_save_dir, f'{img_id}.npz'), cam=cam, fg_cls=fg_cls)

        # * 可视化CAM。
        if cfg.solver.viz_cam and (idx % cfg.solver.viz_step == 0):
            fig.clf()

            pos_names = ['dummy']
            for cls, c in zip(fg_cls, cam, strict=True):
                pos_names.append(f'{cfg.model.fg_names[cls]} {c.min():.1e} {c.max():.1e}')

            viz_cam(fig=fig,
                    img_id=img_id, img=ori_img, label=ori_lb,
                    cls_in_label=np.ones(len(pos_names), dtype=np.uint8),
                    cam=cam,
                    cls_names=pos_names,
                    get_row_col=col_all)

            if args.show_viz:
                fig.show()

            # ** 保存CAM可视化结果。
            fig.savefig(osp.join(cam_viz_dir, f'{img_id}.png'), bbox_inches='tight')

        # * 可视化score。
        if cfg.solver.viz_score and (idx % cfg.solver.viz_step == 0):
            fig.clf()

            score = np.maximum(cam, 0)
            score = min_max_norm(score, dim=(1, 2))

            pos_names = ['dummy']
            for cls, logit in zip(fg_cls, fg_logit, strict=True):
                pos_names.append(f'{cfg.model.fg_names[cls]} {logit:.1f}')

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

        # * 增加处理计数。
        idx += 1

# * 如果只需要eval，此时即可eval并退出。
if cfg.eval.enabled:
    search_and_eval()
