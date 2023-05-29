#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/18 21:10
@File    : max_iou_imp2.py
@Software: PyCharm
@Desc    : 
"""
import torch

from libs.sam.custom_sam.sam_auto import SamAuto


class MaxIoU_IMP2(object):
    def __init__(self,
                 sam_seg_occupied_by_fg_thresh=0.5,
                 fg_occupied_by_sam_seg_thresh: float=0.85,
                 use_seed_when_no_sam: bool=True):
        self.sam_seg_occupied_by_fg_thresh = sam_seg_occupied_by_fg_thresh
        self.fg_occupied_by_sam_seg_thresh = fg_occupied_by_sam_seg_thresh

        self.use_seed_when_no_sam = use_seed_when_no_sam

    def vote(self, seed: torch.Tensor, anns: list[dict[str, ...]]) -> torch.Tensor:
        sam_segs = [torch.as_tensor(SamAuto.decode_segmentation(ann), device=seed.device) for ann in anns]
        bt_sam_segs = torch.stack(sam_segs, dim=0)
        sam_seg_areas = torch.sum(bt_sam_segs, dim=(1, 2), dtype=torch.long)  # 每个标注的面积。
        sam_seged = torch.any(bt_sam_segs, dim=0)  # 有标注的区域。

        segs_cls = torch.full((len(sam_segs),), -1, dtype=torch.long, device=seed.device)  # 记录已经分配标签的标注。
        refined_seed = torch.zeros_like(seed)

        # * 借助SAM，优化每个前景类别的预测。
        for cls in torch.unique(seed):
            if cls == 0 or cls == 255:
                continue

            # * 找到当前类别前景。
            cls_fg = seed == cls
            cls_fg_area = torch.sum(cls_fg, dtype=torch.long)

            # * 遍历SAM的所有标注。
            candidates = []
            for idx, (sam_seg, sam_seg_area) in enumerate(zip(sam_segs, sam_seg_areas, strict=True)):
                # * 若该标注已经赋值，跳过。
                if segs_cls[idx] != -1:
                    continue

                # * 计算标注与当前前景的交集面积。
                intersection_area = torch.sum(cls_fg & sam_seg, dtype=torch.long)

                # * 计算占比。
                sam_seg_occupied_by_fg_prop = intersection_area / sam_seg_area
                fg_occupied_by_sam_seg_prop = intersection_area / cls_fg_area

                # * 判断标注是否可以被分类到当前类别。
                # 若标注满足以下任意一条：
                # 1. 与当前类别的预测交集面积超过自身的X%。
                # 2. 交集占当前类别预测的比例超过Y%。
                # 则将该标注作为当前类别的预测。
                if (sam_seg_occupied_by_fg_prop >= self.sam_seg_occupied_by_fg_thresh) or \
                        (fg_occupied_by_sam_seg_prop >= self.fg_occupied_by_sam_seg_thresh):
                    candidates.append(sam_seg)
                    segs_cls[idx] = cls

            # * SAM没有预测的位置，采用原始预测。
            if self.use_seed_when_no_sam:
                cls_fg_at_no_sam = (~sam_seged) & cls_fg
                candidates.append(cls_fg_at_no_sam)

            # * 将候选位置赋值为当前类别。
            if len(candidates) > 0:
                refined_seed[torch.any(torch.stack(candidates, dim=0), dim=0)] = cls

        return refined_seed
