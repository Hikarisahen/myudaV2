# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T


class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        
        # === [Fix] Remap COCO category IDs to 0 to match model output ===
        # The model is trained to predict class 0. If the dataset has class 1 (or others),
        # evaluation will fail because predictions (0) won't match GT (1).
        # We force the single category to be 0 in the COCO object used for evaluation.
        cat_ids = self.coco.getCatIds()
        if len(cat_ids) == 1 and cat_ids[0] != 0:
            print(f"Remapping COCO category ID from {cat_ids[0]} to 0 for evaluation alignment.")
            old_id = cat_ids[0]
            new_id = 0
            
            # Update categories
            for cat in self.coco.dataset['categories']:
                if cat['id'] == old_id:
                    cat['id'] = new_id
            
            # Update annotations
            for ann in self.coco.dataset['annotations']:
                if ann['category_id'] == old_id:
                    ann['category_id'] = new_id
            
            # Rebuild index
            self.coco.createIndex()
        # ================================================================

        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        # === [关键修改] TIF 格式兼容性处理 ===
        # TIF 可能是 CMYK, I;16, 或者 RGBA，Deformable DETR 需要 RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # ===================================
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # WHU 单类：全部映射成 0（building）
        classes = torch.zeros((len(anno),), dtype=torch.int64)
        assert (classes >= 0).all() and (classes < 1).all(), f"Found invalid labels: {classes.unique()}"

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        # === [核心修改] 解析多边形和角点数据 ===
        polys = None
        corners = None
        # 检查是否包含预处理脚本生成的字段 'cor_cls_poly'
        if anno and "cor_cls_poly" in anno[0]:
            try:
                # 1. 解析 Segmentation 坐标 (Nx128) -> (Nx64x2)
                # JSON中 segmentation 是 [[x1, y1, x2, y2, ...]]，取 [0] 得到内部列表
                poly_list = [obj["segmentation"][0] for obj in anno]
                
                # 转为 Tensor
                polys = torch.as_tensor(poly_list, dtype=torch.float32).reshape(-1, 64, 2)
                
                # 2. 坐标归一化 [0, 1] (除以图片宽高)
                polys[:, :, 0] /= w
                polys[:, :, 1] /= h
                
                # 3. 解析角点类别
                corner_list = [obj["cor_cls_poly"] for obj in anno]
                corners = torch.as_tensor(corner_list, dtype=torch.float32)
                
            except Exception as e:
                print(f"Error loading polygon data for image_id {image_id}: {e}")
                polys = None
                corners = None
        # ======================================

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]
        # [新增] 过滤多边形数据: 如果 box 被过滤了，对应的 poly 也得过滤
        if polys is not None:
            polys = polys[keep]
            corners = corners[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints
        # [新增] 存入 Target 字典
        if polys is not None:
            target["poly_coords"] = polys       # [N, 64, 2]
            target["corner_labels"] = corners   # [N, 64]
        else:
            # 如果没有检测到多边形（空图），创建一个空的 Tensor
            # 形状必须是 (0, 64, 2) 和 (0, 64)
            target["poly_coords"] = torch.zeros((0, 64, 2), dtype=torch.float32)
            target["corner_labels"] = torch.zeros((0, 64), dtype=torch.float32)

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize(scales, max_size=1333),
            # T.RandomSelect(
            #     T.RandomResize(scales, max_size=1333),
            #     T.Compose([
            #         T.RandomResize([400, 500, 600]),
            #         T.RandomSizeCrop(384, 600),
            #         T.RandomResize(scales, max_size=1333),
            #     ])
            # ),
            normalize,
        ])  
        # # 关闭多尺度和随机裁剪以稳定训练
        # return T.Compose([
        #     T.RandomResize([800], max_size=1333),
        #     normalize,
        # ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    # PATHS = {
    #     "train": (root / "train", root / "annotation" / "train_detr.json"),
    #     "val": (root / "validation", root / "annotation" / "validation_detr.json"),
    # } # WHU 数据集文件结构
    PATHS = {
        "train": (root / "train" / "images", root / "train" / "annotation_detr.json"),
        "val": (root / "val" / "images", root / "val" / "annotation_detr.json"),
    } # CrowdAI 数据集文件结构

    img_folder, ann_file = PATHS[image_set]

    # 兼容非指定组织结构的路径
    if not img_folder.exists() or not ann_file.exists():
        flat_ann = root / "annotation_detr.json"
        if flat_ann.exists():
            img_folder = root
            ann_file = flat_ann
            print(f"[Info] Using flat layout for {image_set}: images at {img_folder}, ann at {ann_file}")
        else:
            missing = []
            if not img_folder.exists(): missing.append(str(img_folder))
            if not ann_file.exists(): missing.append(str(ann_file))
            raise FileNotFoundError(f"Missing dataset path(s): {', '.join(missing)}")

    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    return dataset
