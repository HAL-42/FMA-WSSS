#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/8 22:57
@File    : run.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import os
import os.path as osp
import pickle
import shutil
import subprocess
import sys
import traceback
import warnings
from collections import defaultdict

import torch
from alchemy_cat.py_tools import get_local_time_str, gprint, yprint, meow, set_rand_seed, file_md5, rprint
from alchemy_cat.torch_tools import init_env, MovingAverageValueTracker, update_model_state_dict, RNGCacher
from torch.cuda import amp
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path = ['.', './src'] + sys.path  # noqa: E402

from utils.lr_scheduler import SequentialLR
from utils.inf_loader import inf_loader
from utils.loss_items import cal_loss_items


def infer():
    gprint("\n================================== Inference ===================================")
    torch.cuda.empty_cache()  # 尽量释放显存给推理。
    infer_cfg = cfg.infer.cfg.unfreeze()
    infer_cfg.model.resume_file = osp.join(model_save_dir, 'final.pth')

    cfg_pkl = osp.join(cfg.rslt_dir.replace('experiment/', 'configs/'),
                       'infer', f'final', 'cfg.pkl')
    os.makedirs(osp.dirname(cfg_pkl), exist_ok=True)

    with open(cfg_pkl, 'wb') as pkl_f:
        pickle.dump(infer_cfg, pkl_f)
    subprocess.run([sys.executable, 'src/tasks/infer_cam/run.py',
                    '-e', f'{args.eval_only}',
                    '--no_cache', f'{args.no_cache}',
                    '-c', cfg_pkl], check=False)
    gprint("\n================================ Inference End =================================")


# * 读取命令行参数。
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config')
parser.add_argument('-f', '--prefetch_factor', default=2, type=int)
parser.add_argument('-p', '--pin_memory', default=0, type=int)
parser.add_argument("-b", '--benchmark', default=0, type=int)
parser.add_argument("-d", '--is_debug', default=0, type=int)
parser.add_argument("-i", '--infer_only', default=0, type=int)
parser.add_argument("-e", '--eval_only', default=0, type=int)
parser.add_argument('--no_cache', default=0, type=int)
args = parser.parse_args()

# * 初始化环境。
device, cfg = init_env(is_cuda=True,
                       is_benchmark=bool(args.benchmark),
                       is_train=True,
                       config_path=args.config,
                       experiments_root="experiment",
                       rand_seed=True,
                       cv2_num_threads=0,
                       verbosity=True,
                       log_stdout=True,
                       reproducibility=False,
                       is_debug=bool(args.is_debug))

# * 配置路径。
os.makedirs(model_save_dir := osp.join(cfg.rslt_dir, 'checkpoints'), exist_ok=True)

# * 若只需要推理，立即推理并退出。
if args.infer_only:
    infer()
    exit(0)

# * 拷贝随机参考。
for copy_name, (src, dst) in cfg.rand_ref.rand_copy.items():
    print(f"执行随机参考拷贝{copy_name}：\n"
          f"{src}\n"
          f"-->\n"
          f"{dst}")
    if osp.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        os.makedirs(osp.dirname(dst), exist_ok=True)
        if osp.isfile(dst):
            os.remove(dst)
        os.symlink(osp.join(os.getcwd(), src), dst)
print("\n")

# * 数据集。
train_dt = cfg.dt.few_shot.dt
print(train_dt, end="\n\n")

# * 训练数据增强器。
train_auger = cfg.auger.train.cls(train_dt, **cfg.auger.train.ini)
print(train_auger, end="\n\n")

# * 数据加载器。
train_sampler = RandomSampler(train_auger, generator=torch.Generator().manual_seed(cfg.rand_seed))  # 采样的随机独立。
train_loader = DataLoader(train_auger,
                          batch_size=cfg.loader.train.batch_size // cfg.loader.train.sub_iter_num,
                          sampler=train_sampler,
                          num_workers=cfg.loader.num_workers,
                          pin_memory=bool(args.pin_memory),
                          drop_last=True,
                          generator=torch.Generator().manual_seed(cfg.rand_seed),
                          prefetch_factor=args.prefetch_factor
                          )
print(train_sampler)
print(train_loader, end="\n\n")
inf_train_loader = inf_loader(train_loader)

# * 分类模型。
model, get_state, get_named_param_groups = cfg.model.cls(**cfg.model.ini)
cal_model = cfg.model.cal

print(f"以 {cfg.model.initialize_seed} 为随机种子初始化模型。")
model.initialize(seed=cfg.model.initialize_seed)

gprint(f"{get_local_time_str()}    [开始]: ")
if osp.isfile(start_model := osp.join(model_save_dir, 'start.pth')):  # 若存在，认为是参考，载入。
    print(f"    从 {start_model} 载入模型。")
    update_model_state_dict(model, torch.load(start_model, map_location='cpu'), verbosity=3)
else:
    torch.save(get_state(model), start_model)
    print(f"    将模型保存在{start_model}")

print(f"初始模型的MD5为: \n"
      f"{file_md5(start_model)}")
print(model, end="\n\n")

model.set_mode('train')
model = model.to(device)

named_param_groups = {}  # 用于优化器的参数组。

# * 损失函数。
loss_items = cfg.loss.loss_items
print(loss_items, end="\n\n")
for item_name, loss_item in loss_items.items():
    if isinstance(cri := loss_item.cri, torch.nn.Module):
        cri.to(device)
    if hasattr(cri, 'get_named_param_groups'):
        named_param_groups.update(pg := cri.get_named_param_groups())

# * 优化器。
named_param_groups.update(get_named_param_groups(model, **cfg.opt.get_pg.ini))
opt = torch.optim.SGD(params=list(named_param_groups.values()),
                      lr=cfg.opt.base_lr, momentum=cfg.opt.momentum)
print(opt, end="\n\n")

# * 调整学习率。
warm_sched = cfg.sched.warm.cls(opt, **cfg.sched.warm.ini)
main_sched = cfg.sched.main.cls(opt, **cfg.sched.main.ini)
sched = SequentialLR(opt, [warm_sched, main_sched], [cfg.sched.warm.warm_iters])
print(warm_sched)
print(main_sched)
print(sched, end="\n\n")

# * 损失放大器。
scaler = amp.GradScaler(**cfg.amp.scaler.ini)
print(scaler, end="\n\n")

# * 训练前重设随机种子。
set_rand_seed(cfg.rand_seed)
print(f"训练前，重设随机种子为{cfg.rand_seed=}", end="\n\n")

# * 保存或恢复训练前随机状态。实测CLIP前后向+取数据不会改变、也不依赖随机状态，其实无所谓。
cacher = RNGCacher()
cacher.resume_except_cache_to_file(cfg.rslt_dir)
print("\n")

# * Loss跟踪。
loss_trackers = defaultdict(lambda: MovingAverageValueTracker(cfg.solver.loss_average_step *  # 窗长要乘以子迭代次数。
                                                              cfg.loader.train.sub_iter_num))
print(loss_trackers, end="\n\n")

# * Tensorboard。
meow.writer = writer = SummaryWriter(osp.join(cfg.rslt_dir, 'summary'), purge_step=0)

for iteration in tqdm(range(cfg.solver.max_iter), dynamic_ncols=True,
                      desc='训练', unit='批次', miniters=cfg.solver.display_step, bar_format='{l_bar}{bar}{r_bar}\n'):
    meow.iteration = iteration  # 在全局变量中记录当前迭代次数。

    # * 清除之前的梯度。
    opt.zero_grad(set_to_none=True)

    out_of_memory: bool = False
    for sub_iteration in range(cfg.loader.train.sub_iter_num):  # 遍历所有子迭代。
        # * 获取新一个批次数据。
        inp = next(inf_train_loader)
        inp = cfg.io.update_inp(inp)

        try:
            with amp.autocast(enabled=cfg.amp.enabled):
                # * 前向。
                out = cal_model(model, inp)
                out = cfg.io.update_out(inp, out)

                # * 计算损失。
                losses = cal_loss_items(loss_items, inp, out)

            # * 记录损失。
            for loss_name, loss_val in losses.items():  # 记录没有平均过的子迭代损失，配合扩展后的窗长，可得正确结果。
                loss_trackers[loss_name].update(loss_val.item())

            # * 后向。
            scaler.scale(losses['total_loss'] / cfg.loader.train.sub_iter_num).backward()
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                rprint(f"训练时发生CUDA out of memory，将跳过该批次。")
                print(traceback.format_exc())
                out_of_memory = True
            else:
                raise e

    if not out_of_memory:  # 若没有发生CUDA out of memory，则更新参数。
        # * 步进：优化模型参数，并调整学习率。
        scaler.step(opt)

        scale = scaler.get_scale()  # 获取当前缩放因子。
        scaler.update()
        if scale > scaler.get_scale():  # 若缩放因子变小，说明此前更新无效。
            yprint(f"{get_local_time_str()}    [{iteration + 1}/{cfg.solver.max_iter}]: ")
            print(f"    当前{scale=}导致溢出，该次迭代无效。")

    # * 调整学习率。
    sched.step()

    # * 显示训练状态。
    if (iteration + 1) % cfg.solver.display_step == 0:
        # ** 打印时间。
        gprint(f"{get_local_time_str()}    [{iteration + 1}/{cfg.solver.max_iter}]: ")
        # ** 显示损失，并记录到Tensorboard。
        print("    损失：")
        for loss_name, tracker in loss_trackers.items():
            print(f"    {loss_name}: {tracker.mean}")
            writer.add_scalar(f'loss/{loss_name}', tracker.mean, iteration + 1)
        # ** 显示学习率，并记录到Tensorboard。
        print("    学习率：")
        for group_name, group in zip(named_param_groups.keys(), opt.param_groups, strict=True):
            print(f"    lr/{group_name}: {group['lr']}")
            writer.add_scalar(f'optim/lr/{group_name}', group['lr'], iteration + 1)
        # ** 显示损失缩放因子，并记录到Tensorboard。
        if cfg.amp.enabled:
            print(f"    损失缩放因子：\n"
                  f"    scale: {scale}")
            writer.add_scalar('optim/scale', scale, iteration + 1)
        # ** 显示loss中间量。
        print("    中间量：")
        for item_name, loss_item in loss_items.items():
            if hasattr(cri := loss_item.cri, 'display'):
                print(f"    {item_name}:")
                cri.display(iteration, writer)
        # ** 将本显示周期的显存峰值记录到Tensorboard。
        writer.add_scalar(f"gpu/max_memory_allocated",
                          torch.cuda.max_memory_allocated() / 1024 ** 3, iteration + 1)
        writer.add_scalar(f"gpu/max_memory_reserved",
                          torch.cuda.max_memory_reserved() / 1024 ** 3, iteration + 1)
        torch.cuda.reset_peak_memory_stats()

    # * 保存训练中间模型。
    if (iteration + 1) % cfg.solver.save_step == 0:
        torch.save(get_state(model), model_file := osp.join(model_save_dir, f'iter-{iteration + 1}.pth'))
        gprint(f"{get_local_time_str()}    [{iteration + 1}/{cfg.solver.max_iter}]: ")
        print(f"    将模型保存在{model_file}")

    # * 验证性能。
    if (iteration + 1) % cfg.solver.val_step == 0:
        gprint("\n================================== Validation ==================================")
        torch.cuda.empty_cache()  # 尽量释放显存给推理。
        val_cfg = cfg.val.cfg.unfreeze()
        val_cfg.model.resume_file = osp.join(model_save_dir, f'iter-{iteration + 1}.pth')

        cfg_pkl = osp.join(cfg.rslt_dir.replace('experiment/', 'configs/'),
                           'val', f'iter-{iteration + 1}', 'cfg.pkl')
        os.makedirs(osp.dirname(cfg_pkl), exist_ok=True)

        with open(cfg_pkl, 'wb') as pkl_f:
            pickle.dump(val_cfg, pkl_f)
        subprocess.run([sys.executable, 'src/tasks/infer_cam/run.py',
                        '-c', cfg_pkl], check=False)
        try:
            with open(osp.join(cfg.rslt_dir, 'val', f'iter-{iteration + 1}', 'eval', 'statistics.pkl'), 'rb') as pkl_f:
                metrics = pickle.load(pkl_f)
        except Exception:
            warnings.warn("无法读取验证性能统计数据。")
        else:
            writer.add_scalar(f"val/mIoU", metrics['mIoU'], iteration + 1)
        gprint("\n================================ Validation End ================================")

# * 关闭Tensorboard，保存最终模型。
writer.close()

torch.save(get_state(model), model_file := osp.join(model_save_dir, 'final.pth'))
gprint(f"{get_local_time_str()}    [完成]:")
print(f"    将模型保存在{model_file}")

# * 推理最终模型。
infer()
