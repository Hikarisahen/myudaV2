import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np


def guess_img_root(ann_path: Path) -> Path:
    cand = ann_path.parent.parent / "images"
    if cand.exists():
        return cand
    return ann_path.parent


def load_coco(ann_path: Path):
    with ann_path.open("r") as f:
        data = json.load(f)
    images = {img["id"]: img for img in data.get("images", [])}
    anns_per_image = {}
    for ann in data.get("annotations", []):
        anns_per_image.setdefault(ann["image_id"], []).append(ann)
    categories = {c["id"]: c.get("name", str(c["id"])) for c in data.get("categories", [])}
    return images, anns_per_image, categories


def draw_sample(img_bgr, anns, draw_box=True, draw_poly=True, draw_corners=True, thickness=2):
    out = img_bgr.copy()
    for ann in anns:
        # bbox
        if draw_box and "bbox" in ann:
            x, y, w, h = ann["bbox"]
            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))
            cv2.rectangle(out, pt1, pt2, (0, 255, 255), thickness)
        # polygon
        poly = None
        if draw_poly and "segmentation" in ann:
            seg = ann["segmentation"]
            if isinstance(seg, list) and len(seg) > 0 and len(seg[0]) >= 6:
                poly = np.array(seg[0], dtype=np.float32).reshape(-1, 2)
                cv2.polylines(out, [poly.astype(np.int32)], True, (0, 255, 0), thickness, cv2.LINE_AA)
        # corners
        if draw_corners and poly is not None:
            cor = ann.get("cor_cls_poly", None)
            if cor is not None and len(cor) == len(poly):
                cor_mask = np.array(cor) > 0.5
                pts = poly[cor_mask].astype(np.int32)
                for (px, py) in pts:
                    cv2.circle(out, (int(px), int(py)), 2, (0, 0, 255), -1)
    return out


def main():
    parser = argparse.ArgumentParser(description="Visualize COCO-style polygons/crops")
    parser.add_argument("--ann", required=True, help="Path to annotation json")
    parser.add_argument("--img_root", default=None, help="Root directory of images (defaults to ../images next to ann)")
    parser.add_argument("--output_dir", default="./vis_out/vis_crop", help="Where to save visualizations")
    parser.add_argument("--max_images", type=int, default=20, help="How many images to visualize")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before picking")
    parser.add_argument("--seed", type=int, default=42, help="Random seed when shuffling")
    parser.add_argument("--draw_box", action="store_true", help="Draw bounding boxes")
    parser.add_argument("--no_draw_box", dest="draw_box", action="store_false")
    parser.add_argument("--draw_poly", action="store_true", help="Draw polygons")
    parser.add_argument("--no_draw_poly", dest="draw_poly", action="store_false")
    parser.add_argument("--draw_corners", action="store_true", help="Draw corner points when cor_cls_poly exists")
    parser.add_argument("--no_draw_corners", dest="draw_corners", action="store_false")
    parser.set_defaults(draw_box=True, draw_poly=True, draw_corners=True)
    args = parser.parse_args()

    ann_path = Path(args.ann)
    img_root = Path(args.img_root) if args.img_root else guess_img_root(ann_path)
    if not img_root.exists():
        raise FileNotFoundError(f"Image root not found: {img_root}")

    images, anns_per_image, _ = load_coco(ann_path)

    image_ids = list(images.keys())
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(image_ids)
    image_ids = image_ids[: args.max_images]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_id in image_ids:
        info = images[img_id]
        file_name = info["file_name"]
        img_path = img_root / file_name
        if not img_path.exists():
            print(f"Skip missing image: {img_path}")
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Fail to read image: {img_path}")
            continue
        anns = anns_per_image.get(img_id, [])
        vis = draw_sample(img, anns, draw_box=args.draw_box, draw_poly=args.draw_poly, draw_corners=args.draw_corners)
        save_path = out_dir / f"{Path(file_name).stem}_vis.png"
        cv2.imwrite(str(save_path), vis)
        print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
    """
    用于可视化 COCO 格式的多边形标注和裁剪区域。
    python util/vis_crop.py \
        --ann /data/zfx/datasets/WHUuda/train/annotation_detr.json \
        --img_root /data/zfx/datasets/WHUuda/train/images \
        --output_dir vis_out/vis_crop \
        --max_images 20 --shuffle
    """
