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
import multiprocessing as mp
import os
import os.path as osp
import sys

import matplotlib
import numpy as np
import torch
from alchemy_cat.acplot import BGR2RGB, col_all
from alchemy_cat.contrib.tasks.wsss.viz import viz_cam
from alchemy_cat.py_tools import rprint
from alchemy_cat.torch_tools import init_env, update_model_state_dict
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path = ['.', './src'] + sys.path  # noqa: E402

from utils.cache_dir import CacheDir
from libs.seeding.score import cam2score
from libs.seeding.aff.tools import merge_att
from utils.eval_cams import search_and_eval
from utils.resize import resize_cam

mp.set_start_method('spawn', force=True)

if __name__ == '__main__':
    # * 读取命令行参数。
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    parser.add_argument('-f', '--prefetch_factor', default=2, type=int)
    parser.add_argument('-p', '--pin_memory', default=0, type=int)
    parser.add_argument("-b", '--benchmark', default=0, type=int)
    parser.add_argument("-d", '--is_debug', default=0, type=int)
    parser.add_argument('-s', '--show_viz', default=0, type=int)
    parser.add_argument('-e', '--eval_only', default=0, type=int)
    parser.add_argument('-P', '--pool_size', default=0, type=int)
    parser.add_argument('--no_cache', default=0, type=int)
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
    if not args.show_viz:
        matplotlib.use('Agg')
    print(f"{matplotlib.get_backend()=}")

    # * 配置路径。
    cam_dir = CacheDir(osp.join(cfg.rslt_dir, 'cam'), '/tmp/infer_cam/cam',
                       exist='delete' if not args.eval_only else 'cache',
                       enabled=not bool(args.no_cache))

    if cfg.solver.viz_cam:
        os.makedirs(cam_viz_dir := osp.join(cfg.rslt_dir, 'viz', 'cam'), exist_ok=True)
    if cfg.solver.viz_score:
        os.makedirs(score_viz_dir := osp.join(cfg.rslt_dir, 'viz', 'score'), exist_ok=True)

    # * 如果只需要eval，此时即可eval并退出。
    if args.eval_only:
        assert cfg.eval.enabled
        search_and_eval(cfg.dt.val.dt, cam_dir.at, cfg.eval.seed, cfg.rslt_dir, args.pool_size)
        cam_dir.terminate()
        exit(0)

    # * 数据集。
    val_dt = cfg.dt.val.dt
    print(val_dt, end="\n\n")
    fg_names = val_dt.class_names[1:]

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
    cal_model = cfg.model.cal

    if resume_file := cfg.model.resume_file:
        update_model_state_dict(model, torch.load(resume_file, map_location='cpu'), verbosity=3)
    print(model, end="\n\n")

    model.set_mode('eval')
    model = model.to(device)

    # * 准备可视化图像。
    fig = plt.figure(dpi=600)

    idx = 0  # 用于计数推理了多少张图。

    for inp in tqdm(val_loader, dynamic_ncols=True, desc='推理', unit='批次', miniters=10):
        # * 跳过已有（只对bt=1有效）。
        if len(inp.img_id) == 1:
            id_ = inp.img_id[0]
            if osp.isfile(cam_file := osp.join(cam_dir, f'{id_}.npz')):
                try:
                    _ = np.load(cam_file, allow_pickle=True)
                except Exception as e:
                    rprint(f"[重算] {id_} 存在但无法加载，重新计算。")
                else:
                    print(f"[重算] {id_} 已经存在且可以被正确加载。")
                    idx += 1
                    continue

        # * 获取新一个批次数据。
        inp = cfg.io.update_inp(inp)

        # * 前向。
        out = cal_model(model, inp)
        out = cfg.io.update_out(inp, out)

        # * 获取out中正类CAM pos_cam，转到CPU上，随后按照他们的batch_idx分组。
        batch_size = inp.img.shape[0]

        pos_batch_idx = np.nonzero(fg_cls_lb := inp.fg_cls_lb.cpu().numpy())[0]
        pos_cam = out.pos_cam.to(dtype=torch.float32, device='cpu').numpy()  # PHW，CPU上诸多操作不支持FP16，故转为FP32。
        sample_cam = [pos_cam[pos_batch_idx == idx, :, :] for idx in range(batch_size)]  # [样本数xHxW]

        sample_fg_cls = [np.nonzero(fg_cls_lb[i, :])[0] for i in range(batch_size)]

        fg_logits = out.fg_logits.detach().to(dtype=torch.float32, device='cpu').numpy()
        sample_fg_logit = [fg_logits[i, fg_cls_lb[i, :].astype(bool)] for i in range(batch_size)]

        sample_att = torch.stack(out.att_weights, dim=1).detach().to(torch.float32)  # [样本数xDxLxL]

        # * 遍历每张图的id和CAM，保存到文件并可视化之。
        for img_id, cam, fg_cls, fg_logit, att, all_logit in zip(inp.img_id,
                                                                 sample_cam, sample_fg_cls, sample_fg_logit,
                                                                 sample_att, fg_logits,
                                                                 strict=True):
            # * 将CAM转到原始图像的尺寸上。✖放弃，改为储存原始尺寸的CAM。
            # cam = cv2.resize(cam.transpose(1, 2, 0), (ori_w, ori_h))
            # if cam.ndim == 2:
            #     cam = cam[None, :, :]
            # else:
            #     cam = cam.transpose(2, 0, 1)
            # cam = F.interpolate(cam.unsqueeze(0), size=(ori_h, ori_w), mode='bilinear',
            #                     align_corners=False).squeeze(0))  # [样本数xHxW]
            # cam = cam.numpy().astype(np.float16)

            # * 保存CAM和有关中间量。
            saved = dict(cam=cam, fg_cls=fg_cls, fg_logit=fg_logit, all_logit=all_logit)
            if cfg.solver.save_att:
                att = merge_att(att, cfg.solver.save_att).to(dtype=torch.float16, device='cpu').numpy()  # FP16节省空间。
                saved['att'] = att
            np.savez(osp.join(cam_dir, f'{img_id}.npz'), **saved)

            # * 如果需要可视化，根据img_id获取原始图像。
            if (cfg.solver.viz_cam or cfg.solver.viz_score) and (idx % cfg.solver.viz_step == 0):
                ori_inp = val_dt.get_by_img_id(img_id)
                ori_img, ori_lb = BGR2RGB(ori_inp.img), ori_inp.lb
                ori_h, ori_w = ori_img.shape[:2]

            # * 可视化CAM。
            if cfg.solver.viz_cam and (idx % cfg.solver.viz_step == 0):
                fig.clf()

                pos_names = ['dummy']
                for cls, c in zip(fg_cls, cam, strict=True):
                    pos_names.append(f'{fg_names[cls]} {c.min():.1e} {c.max():.1e}')

                resized_cam = resize_cam(cam, (ori_h, ori_w))

                viz_cam(fig=fig,
                        img_id=img_id, img=ori_img, label=ori_lb,
                        cls_in_label=np.ones(len(pos_names), dtype=np.uint8),
                        cam=resized_cam,
                        cls_names=pos_names,
                        get_row_col=col_all)

                if args.show_viz:
                    fig.show()

                # ** 保存CAM可视化结果。
                fig.savefig(osp.join(cam_viz_dir, f'{img_id}.png'), bbox_inches='tight')

            # * 可视化score。
            if cfg.solver.viz_score and (idx % cfg.solver.viz_step == 0):
                fig.clf()

                score = cam2score(cam, (ori_h, ori_w), cfg.solver.viz_score.resize_first)

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

            # * 增加处理计数。
            idx += 1

    # * 如果需要保存CAM，则将暂存的cam_affed拷贝到保存目录。
    if cfg.solver.save_cam:
        cam_dir.flush()

    # * 如果只需要eval，此时即可eval并退出。
    if cfg.eval.enabled:
        torch.cuda.empty_cache()
        search_and_eval(cfg.dt.val.dt, cam_dir.at, cfg.eval.seed, cfg.rslt_dir, args.pool_size)

    # * 则删除CAM缓存目录。
    cam_dir.terminate()
