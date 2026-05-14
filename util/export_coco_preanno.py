"""
将模型预测导出为 COCO 格式的预标注 JSON，便于在 Label Studio / X-AnyLabeling / CVAT
等标注工具中作为 pre-annotation 导入，由人工修正后形成最终评估集。

用法示例：
    python -m util.export_coco_preanno \
        --checkpoint weight/checkpoint_wuhan.pth \
        --image_dir /home/zfx/datasets/wuhan/eval_subset/images \
        --output_json /home/zfx/datasets/wuhan/eval_subset/preanno.json \
        --with_box_refine --num_queries 300 --num_feature_levels 4 \
        --dataset_file coco --source_path "" --target_path "" \
        --score_thresh 0.3 --corner_thresh 0.45 --onlycorner --enable_nms
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# 允许从子目录运行：把项目根加入 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from main import get_args_parser
from datasets.coco import make_coco_transforms
from util.misc import nested_tensor_from_tensor_list
from infer_visualize import (
    load_model,
    collect_images,
    apply_nms,
    repair_polygon_self_intersection,
    polygon_nms_indices,
)


def polygon_to_bbox(poly):
    x_min = float(np.min(poly[:, 0]))
    y_min = float(np.min(poly[:, 1]))
    x_max = float(np.max(poly[:, 0]))
    y_max = float(np.max(poly[:, 1]))
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def polygon_area(poly):
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def predict_polygons(
    model,
    img_path,
    transform,
    device,
    score_thresh,
    corner_thresh,
    only_corner,
    enable_nms,
    nms_thresh,
    enable_polygon_nms,
    polygon_nms_iou,
    polygon_nms_downsample,
    repair_self_intersection,
):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    img_t, _ = transform(img, None)
    samples = nested_tensor_from_tensor_list([img_t]).to(device)

    with torch.no_grad():
        outputs = model(samples)

    pred_logits = outputs["pred_logits"][0]
    pred_polys_batch = outputs.get(
        "pred_polys_evolve_1",
        outputs.get("pred_polys_init", outputs.get("pred_polys")),
    )
    pred_polys = pred_polys_batch[0]

    pred_corners_logits_batch = outputs.get(
        "pred_vtx_logits_evolve_1",
        outputs.get("pred_corners_init", outputs.get("pred_corners")),
    )
    pred_corners_logits = pred_corners_logits_batch[0]
    pred_corners = pred_corners_logits.sigmoid()

    probs = pred_logits.sigmoid()
    if probs.shape[-1] == 1:
        scores = probs.squeeze(-1)
    else:
        scores, _ = probs[..., :-1].max(-1)

    keep = scores > score_thresh
    polys = pred_polys[keep].cpu().numpy()
    corners = pred_corners[keep].cpu().numpy()
    det_scores = scores[keep].cpu().numpy()

    polys[..., 0] *= w
    polys[..., 1] *= h

    out_polys, out_scores = [], []
    for poly, corner, det_score in zip(polys, corners, det_scores):
        is_corner = (
            apply_nms(poly, corner, corner_thresh, nms_thresh)
            if enable_nms
            else (corner > corner_thresh)
        )
        if only_corner and is_corner.sum() >= 3:
            kept_poly = poly[is_corner]
        else:
            kept_poly = poly
        if repair_self_intersection:
            kept_poly = repair_polygon_self_intersection(kept_poly)
        if kept_poly.shape[0] < 3:
            continue
        out_polys.append(kept_poly)
        out_scores.append(float(det_score))

    if enable_polygon_nms and len(out_polys) > 1:
        keep_idx = polygon_nms_indices(
            out_polys,
            out_scores,
            h,
            w,
            iou_thresh=polygon_nms_iou,
            downsample=polygon_nms_downsample,
        )
        out_polys = [out_polys[i] for i in keep_idx]
        out_scores = [out_scores[i] for i in keep_idx]

    return out_polys, out_scores, w, h


def build_coco(
    img_paths,
    model,
    transform,
    device,
    args,
    image_path_mode,
    image_root,
    category_id,
    category_name,
):
    images, annotations = [], []
    ann_id = 1
    for image_id, p in enumerate(img_paths, start=1):
        polys, scores, w, h = predict_polygons(
            model,
            p,
            transform,
            device,
            score_thresh=args.score_thresh,
            corner_thresh=args.corner_thresh,
            only_corner=args.onlycorner,
            enable_nms=args.enable_nms,
            nms_thresh=args.nms_thresh,
            enable_polygon_nms=not args.disable_polygon_nms,
            polygon_nms_iou=args.polygon_nms_iou,
            polygon_nms_downsample=args.polygon_nms_downsample,
            repair_self_intersection=not args.disable_self_intersection_repair,
        )

        if image_path_mode == "basename":
            file_name = os.path.basename(p)
        elif image_path_mode == "relative" and image_root:
            file_name = os.path.relpath(p, image_root).replace("\\", "/")
        else:
            file_name = p.replace("\\", "/")

        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": w,
                "height": h,
            }
        )

        for poly, score in zip(polys, scores):
            seg = [float(c) for c in poly.reshape(-1).tolist()]
            bbox = polygon_to_bbox(poly)
            area = polygon_area(poly)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [seg],
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "score": score,
                }
            )
            ann_id += 1

        print(f"[{image_id}/{len(img_paths)}] {os.path.basename(p)}: {len(polys)} instances")

    coco = {
        "info": {
            "description": "Model pre-annotation for manual correction",
            "source": "myudaV2 export_coco_preanno.py",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": category_id, "name": category_name, "supercategory": "building"}],
    }
    return coco


def main():
    parser = get_args_parser()
    for action in parser._actions:
        if action.dest in {"source_path", "target_path"}:
            action.required = False
            action.default = ""

    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pth")
    parser.add_argument("--image_path", help="Single image for inference")
    parser.add_argument("--image_dir", help="Folder containing images for batch inference")
    parser.add_argument("--output_json", required=True, help="Output COCO JSON path")
    parser.add_argument("--image_path_mode", choices=["basename", "relative", "absolute"], default="basename",
                        help="How to write file_name in COCO images field")
    parser.add_argument("--image_root", default=None, help="Used when --image_path_mode relative")
    parser.add_argument("--category_id", type=int, default=1, help="Category id in output COCO (1 is most tool-friendly)")
    parser.add_argument("--category_name", type=str, default="building")
    parser.add_argument("--score_thresh", type=float, default=0.3,
                        help="过滤低置信度预测；预标注阶段建议比可视化稍低（0.25~0.35），减少漏检")
    parser.add_argument("--corner_thresh", type=float, default=0.45)
    parser.add_argument("--onlycorner", action="store_true")
    parser.add_argument("--enable_nms", action="store_true")
    parser.add_argument("--nms_thresh", type=float, default=10.0)
    parser.add_argument("--disable_polygon_nms", action="store_true")
    parser.add_argument("--polygon_nms_iou", type=float, default=0.3)
    parser.add_argument("--polygon_nms_downsample", type=int, default=4)
    parser.add_argument("--disable_self_intersection_repair", action="store_true")

    args = parser.parse_args()

    if not args.image_path and not args.image_dir:
        raise ValueError("Please provide --image_path or --image_dir")

    args.device = torch.device(args.device)
    model = load_model(args, args.checkpoint)
    transform = make_coco_transforms("val")

    img_paths = collect_images(args.image_path, args.image_dir)
    if not img_paths:
        raise ValueError("No images found with given path(s)")

    image_root = args.image_root or args.image_dir
    coco = build_coco(
        img_paths,
        model,
        transform,
        args.device,
        args,
        image_path_mode=args.image_path_mode,
        image_root=image_root,
        category_id=args.category_id,
        category_name=args.category_name,
    )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)
    print(f"\nSaved COCO pre-annotation to: {out_path}")
    print(f"  images: {len(coco['images'])}, annotations: {len(coco['annotations'])}")


if __name__ == "__main__":
    main()
