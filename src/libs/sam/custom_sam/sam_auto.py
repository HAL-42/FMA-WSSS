#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/16 20:04
@File    : sam_auto.py
@Software: PyCharm
@Desc    : 
"""
from copy import deepcopy
from typing import Optional, List, Iterable, Tuple, Dict, Any

import numpy as np
import torch
from alchemy_cat.py_tools import ADict
from torchvision.ops import batched_nms

from segment_anything import SamAutomaticMaskGenerator
from segment_anything.modeling import Sam
from segment_anything.utils.amg import (MaskData, calculate_stability_score, batched_mask_to_box,
                                        is_box_near_crop_edge, uncrop_masks, mask_to_rle_pytorch, batch_iterator,
                                        uncrop_boxes_xyxy, uncrop_points, coco_encode_rle, rle_to_mask,
                                        box_xyxy_to_xywh, area_from_rle)

__all__ = ["SamAnns", "SamAuto"]


class SamAnns(list):
    """SAM返回结果类。继承自list，但是可以通过.访问到纵向属性。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = ADict()

    def clear_data(self):
        self.data = ADict()

    def stack_data_like(self, anns: 'SamAnns'):
        like_data = anns.data

        self.clear_data()
        self.stack_data(masks='masks' in like_data,
                        areas='areas' in like_data,
                        device=like_data.device)

    def stack_data(self, masks: bool=True, areas: bool=True, device=torch.device('cpu')):
        """将data在纵向维度stack。

        Args:
            masks: 若为True，则将masks stack到self.data.masks中。
            areas: 若为True，则将area stack到self.data.areas中。
            device: stack所得的tensor的device。

        Returns:
            自身。
        """
        self.data.device = torch.device(device)

        if masks:
            m = [torch.as_tensor(SamAuto.decode_segmentation(ann, replace=False), dtype=torch.bool, device=device)
                 for ann in self]
            self.data.masks = torch.stack(m, dim=0)

        if areas:
            self.data.areas = torch.as_tensor([ann['area'] for ann in self], dtype=torch.int32, device=device)

    def add_item(self, key: str, val: Iterable):
        """给所有标注增加一个名为key的项目。

        Args:
            key: 键名。
            val: 键值，可迭代，长度为标注数目相同。
        """
        for ann, v in zip(self, val, strict=True):
            ann[key] = v


class SamAuto(SamAutomaticMaskGenerator):
    level_num = 3  # level数目。

    def __init__(self,
                 model: Sam,
                 points_per_side: Optional[int] = 32,
                 points_per_batch: int = 64,
                 pred_iou_thresh: float = 0.88,
                 stability_score_thresh: float = 0.95,
                 stability_score_offset: float = 1.0,
                 box_nms_thresh: float = 0.7,
                 crop_n_layers: int = 0,
                 crop_nms_thresh: float = 0.7,
                 crop_overlap_ratio: float = 512 / 1500,
                 crop_n_points_downscale_factor: int = 1,
                 point_grids: Optional[List[np.ndarray]] = None,
                 min_mask_region_area: int = 0,
                 output_mode: str = "binary_mask",
                 score_thresh_offset: Iterable[float] = (0.0, 0.0, 0.0),
                 score_nms_offset: Iterable[float] = (0.0, 0.0, 0.0),
                 stability_score_bias: Iterable[float] = (0.0, 0.0, 0.0),
                 rock_sand_water_thresh: float=None,
                 rock_sand_water_chunk_size: int=0
                 ) -> None:
        """
        Using a SAM model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM with a ViT-H backbone.

        Arguments:
          model (Sam): The SAM model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crop_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
          score_thresh_offset (tuple(float, float, float)): SAM三种mask（分分、分、总）在阈值时的偏移。
          score_nms_offset (tuple(float, float, float)): SAM三种mask（分分、分、总）在NMS时的偏移。
          stability_score_bias (tuple(float, float, float)): SAM三种mask（分分、分、总）在计算稳定性时的偏移。
          rock_sand_water_thresh: 石沙水过滤的recall阈值。
          rock_sand_water_chunk_size: 石沙水过滤的batch大小。若为0，不做batch。
        """
        super().__init__(model=model,
                         points_per_side=points_per_side,
                         points_per_batch=points_per_batch,
                         pred_iou_thresh=pred_iou_thresh,
                         stability_score_thresh=stability_score_thresh,
                         stability_score_offset=stability_score_offset,
                         box_nms_thresh=box_nms_thresh,
                         crop_n_layers=crop_n_layers,
                         crop_nms_thresh=crop_nms_thresh,
                         crop_overlap_ratio=crop_overlap_ratio,
                         crop_n_points_downscale_factor=crop_n_points_downscale_factor,
                         point_grids=point_grids,
                         min_mask_region_area=min_mask_region_area,
                         output_mode=output_mode)
        self.score_thresh_offset = torch.as_tensor(score_thresh_offset, dtype=torch.float, device=self.predictor.device)
        self.score_nms_offset = torch.as_tensor(score_nms_offset, dtype=torch.float, device=self.predictor.device)
        self.stability_score_bias = torch.as_tensor(stability_score_bias, dtype=torch.float,
                                                    device=self.predictor.device)
        assert self.score_thresh_offset.shape == (3,)
        assert self.score_nms_offset.shape == (3,)
        assert self.stability_score_bias.shape == (3,)

        self.rock_sand_water_thresh = rock_sand_water_thresh
        self.rock_sand_water_chunk_size = rock_sand_water_chunk_size

    @torch.no_grad()
    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.

        Returns:
           list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """
        h, w, _ = image.shape

        # Generate masks
        mask_data = self._generate_masks(image)

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(  # 填洞、削岛。
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )

        # CHANGE 做石沙水过滤。
        if self.rock_sand_water_thresh is not None:
            mask_data = self.rock_sand_water(mask_data)

        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [coco_encode_rle(rle) for rle in mask_data["rles"]]  # coco的压缩游程编码。
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]  # 返回mask。
        else:
            mask_data["segmentations"] = mask_data["rles"]  # 返回未压缩的游程编码。

        # Write mask records
        curr_anns = SamAnns()
        for idx in range(len(mask_data["segmentations"])):  # 以mask为首维，返回。
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "segmentation_mode": self.output_mode,  # CHANGE 增加输出模式。
                "img_hw": (h, w),  # CHANGE 增加图片尺寸。
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),  # box转为xywh格式。
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "predicted_iou_thresh": mask_data["iou_preds_thresh"][idx].item(),  # CHANGE 增加用于NMS和thresh的iou。
                "predicted_iou_nms": mask_data["iou_preds_nms"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
                "stability_score_bias": mask_data["stability_score_bias"][idx].item(),
                "level": mask_data["level"][idx].item(),
            }
            curr_anns.append(ann)

        return curr_anns

    @staticmethod
    def decode_segmentation(ann: dict[str, ...], replace: bool=False) -> np.ndarray:
        match ann['segmentation_mode']:
            case 'coco_rle':
                raise NotImplementedError
            case 'binary_mask':
                ret = ann['segmentation']
            case 'uncompressed_rle':
                ret = rle_to_mask(ann['segmentation'])
            case _:
                raise ValueError(f"Unknown segmentation mode: {ann['segmentation_mode']}")

        if replace:
            ann['segmentation'] = ret
            ann['segmentation_mode'] = 'binary_mask'

        return ret

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        self.predictor.set_image(cropped_im)  # 获取crop内的特征。

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]  # [[w, h]]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale  # 将crop内所有点的相对坐标转为绝对坐标。

        # Generate masks for this crop in batches
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(points, cropped_im_size, crop_box, orig_size)
            data.cat(batch_data)  # 将每个batch的结果拼接起来。
            del batch_data
        self.predictor.reset_image()

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),  # 使用mask的BBox做NMS。
            data["iou_preds_nms"],  # CHANGE BBox分数就是mask（偏移后）的iou预测。
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)  # 将框、点坐标、加上offset，转为原图坐标。
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])  # 记录每个mask源自的crop。

        return data

    def _process_batch(
            self,
            points: np.ndarray,
            im_size: Tuple[int, ...],
            crop_box: List[int],
            orig_size: Tuple[int, ...],
    ) -> MaskData:
        orig_h, orig_w = orig_size

        # Run model on this batch
        transformed_points = self.predictor.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)  # (N, 2)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)  # (N,)
        masks, iou_preds, _ = self.predictor.predict_torch(  # (N, 3, H, W), (N, 3)
            in_points[:, None, :],  # (N, 1, 2)，每批一个前景点。
            in_labels[:, None],  # (N, 1)，全是fg。
            multimask_output=True,
            return_logits=True,
        )

        # CHANGE iou_preds加上偏移量，并记录到MaskData中。
        iou_preds_thresh = iou_preds + self.score_thresh_offset[None, :]
        iou_preds_nms = iou_preds + self.score_nms_offset[None, :]

        # CHANGE 计算稳定度的偏移量。
        stability_score_bias = self.stability_score_bias.expand_as(iou_preds)

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),  # (Nx3, H, W)
            iou_preds=iou_preds.flatten(0, 1),  # (Nx3,)
            iou_preds_thresh=iou_preds_thresh.flatten(0, 1),
            iou_preds_nms=iou_preds_nms.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),  # (N, 3【重复】, 2)
            stability_score_bias=stability_score_bias.flatten(0, 1),
            # CHANGE 记录每个mask的level。
            level=torch.as_tensor([0, 1, 2],
                                  dtype=torch.uint8,
                                  device=iou_preds.device).expand_as(iou_preds).flatten(0, 1)
        )
        del masks

        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds_thresh"] > self.pred_iou_thresh  # CHANGE（偏移后的）预测iou要大于阈值。
            data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(  # mask要稳定。
            data["masks"], self.predictor.model.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            # CHANGE 稳定度加上偏移量，然后和阈值比较。
            keep_mask = (data["stability_score"] + data["stability_score_bias"]) >= self.stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold  # 转为二进制mask。
        data["boxes"] = batched_mask_to_box(data["masks"])  # 计算mask对应的BBox。

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)  # mask加上偏移，对齐原图。
        data["rles"] = mask_to_rle_pytorch(data["masks"])  # 获取mask的未压缩游程编码。
        del data["masks"]

        return data

    @staticmethod
    def _mask_intersection_areas(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        """Computes the intersection area between two masks.

        Args:
            mask1: Binary masks of shape (N1, H, W).
            mask2: Binary masks of shape (N2, H, W).

        Returns:
            A tensor of shape (N1, N2) containing the intersection area between the two masks.
        """
        # * 计算两个mask的交集。
        intersection = mask1[:, None, :, :] & mask2[None, :, :, :]  # (N1, N2, H, W)

        # * 计算交集的面积。
        intersection_areas = intersection.sum(dim=(2, 3), dtype=torch.long)  # (N1, N2)

        return intersection_areas

    @staticmethod
    def _chunk_mask_intersection_areas(mask1: torch.Tensor, mask2: torch.Tensor, chunksize: int=0) -> torch.Tensor:
        """Computes the intersection area between two masks.

        Args:
            mask1: Binary masks of shape (N1, H, W).
            mask2: Binary masks of shape (N2, H, W).
            chunksize: Chunk size to use for the computation.

        Returns:
            A tensor of shape (N1, N2) containing the intersection area between the two masks.
        """
        if chunksize == 0:
            return SamAuto._mask_intersection_areas(mask1, mask2)

        ret = []

        for i in range(0, mask1.shape[0], chunksize):
            chunk = mask1[i:i + chunksize, :, :]  # (chunksize, H, W)
            ret.append(SamAuto._mask_intersection_areas(chunk, mask2))  # (chunksize, N2)

        ret = torch.cat(ret, dim=0)  # (N1, N2)

        return ret

    def rock_sand_water(self, mask_data: MaskData) -> MaskData:
        # * 若没有mask，直接返回。
        if len(mask_data["rles"]) == 0:
            return mask_data

        # * 将mask_data按照level等级拆分。
        levels_mask_datas = []
        for level in list(range(self.level_num))[::-1]:  # 等级从大到小。
            # * 滤除指定等级的mask。
            level_filter = torch.from_numpy(mask_data["level"] == level)
            level_mask_data = deepcopy(mask_data)
            level_mask_data.filter(level_filter)

            # * 计算cuda上的布尔掩码。
            level_mask_data['masks'] = torch.as_tensor(np.array([rle_to_mask(rle) for rle in level_mask_data["rles"]]),
                                                       dtype=torch.bool,
                                                       device=self.predictor.device)

            levels_mask_datas.append(level_mask_data)

        # * 从大到小，对每个等级的mask进行处理。
        rsw_mask_data = None
        for level_mask_data in levels_mask_datas:
            # * 最高级mask作为结果容器。
            if rsw_mask_data is None:
                rsw_mask_data = level_mask_data
                continue

            # * 若结果容器为空，取当前等级mask作为结果容器。
            if len(rsw_mask_data["rles"]) == 0:
                rsw_mask_data = level_mask_data
                continue

            # * 若当前等级mask为空，跳过。
            if len(level_mask_data["rles"]) == 0:
                continue

            # * 计算当前等级二进制掩码与结果容器掩码的交集面积。
            intersection_areas = self._chunk_mask_intersection_areas(level_mask_data['masks'],  # (N1, N2)
                                                                     rsw_mask_data['masks'],
                                                                     chunksize=self.rock_sand_water_chunk_size)

            # * 计算当前等级掩码与结果容器中掩码最大的召回率。
            level_mask_areas = torch.sum(level_mask_data['masks'], dim=(1, 2), dtype=torch.long)  # (N1,)
            level_recalls = intersection_areas.float() / level_mask_areas[:, None]  # (N1, N2)
            level_max_recall = torch.amax(level_recalls, dim=1)  # (N1,)

            # * 当前等级mask，只有其最大召回率小于阈值的，才会被加入结果容器。
            level_filter = level_max_recall < self.rock_sand_water_thresh
            level_mask_data.filter(level_filter)
            rsw_mask_data.cat(level_mask_data)

        # * 删除masks属性，节省显存。
        del rsw_mask_data['masks']

        return rsw_mask_data
