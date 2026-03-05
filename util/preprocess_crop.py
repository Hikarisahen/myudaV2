import argparse
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import rasterio


def linear_stretch(img, percent=2):
    """2%-98% 线性拉伸，16bit -> 8bit。"""
    img_float = img.astype(np.float32)
    lower = np.percentile(img_float, percent)
    upper = np.percentile(img_float, 100 - percent)
    denom = max(upper - lower, 1e-6)
    img_stretched = (img_float - lower) / denom * 255.0
    img_stretched = np.clip(img_stretched, 0, 255).astype(np.uint8)
    return img_stretched


def load_ann_index(ann_path: Path):
    """载入 COCO 标注，返回按 basename 索引的列表以及原始 categories/info/licenses。"""
    with ann_path.open("r") as f:
        data = json.load(f)
    images = data.get("images", [])
    anns = data.get("annotations", [])
    categories = data.get("categories", [])
    info = data.get("info", {})
    licenses = data.get("licenses", [])

    img_by_id = {img["id"]: img for img in images}
    anns_by_base = defaultdict(list)
    for ann in anns:
        img_info = img_by_id.get(ann["image_id"])
        if not img_info:
            continue
        fname = Path(img_info.get("file_name", "")).name
        anns_by_base[fname].append(ann)
    return anns_by_base, categories, info, licenses


def process_annotations_for_crop(anns, scale, x0, y0, crop_size, ann_id_start, image_id, min_object_area):
    """将标注缩放+裁剪到当前 patch，返回新标注列表与下一个 ann_id。"""
    new_anns = []
    ann_id = ann_id_start
    for ann in anns:
        seg = ann.get("segmentation")
        if not seg or len(seg) == 0 or len(seg[0]) < 6:
            continue
        coords = np.array(seg[0], dtype=np.float32).reshape(-1, 2)
        coords *= scale

        bbox = ann.get("bbox", [0, 0, 0, 0])
        bbox = np.array(bbox, dtype=np.float32)
        bbox[:2] *= scale
        bbox[2:] *= scale

        x, y, w, h = bbox.tolist()
        if x + w <= x0 or y + h <= y0 or x >= x0 + crop_size or y >= y0 + crop_size:
            continue

        coords[:, 0] -= x0
        coords[:, 1] -= y0
        coords[:, 0] = np.clip(coords[:, 0], 0, crop_size - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, crop_size - 1)

        if coords.shape[0] < 3:
            continue

        xmin, ymin = coords.min(axis=0)
        xmax, ymax = coords.max(axis=0)
        bw = max(xmax - xmin, 1e-3)
        bh = max(ymax - ymin, 1e-3)

        area = bw * bh
        if area < min_object_area:  # 极小目标直接丢弃，避免噪声
            continue

        new_ann = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": ann["category_id"],
            "segmentation": [coords.reshape(-1).tolist()],
            "bbox": [float(xmin), float(ymin), float(bw), float(bh)],
            "area": float(area),
            "iscrowd": ann.get("iscrowd", 0),
        }
        if "cor_cls_poly" in ann:
            new_ann["cor_cls_poly"] = ann["cor_cls_poly"]

        new_anns.append(new_ann)
        ann_id += 1
    return new_anns, ann_id


def preprocess_image(large_img_path, output_dir, target_scale, crop_size, overlap, anns_for_img=None,
                     ann_id_start=1, image_id_start=1, min_crop_mean=5.0, min_object_area=16.0,
                     save_format="jpg"):
    filename = os.path.basename(large_img_path)
    base_name = os.path.splitext(filename)[0]

    try:
        with rasterio.open(large_img_path) as src:
            print(f"处理 {filename}: 原始 {src.width}x{src.height}, 波段={src.count}, 类型={src.dtypes[0]}")
            img = src.read([1, 2, 3])
            img = np.transpose(img, (1, 2, 0))
    except Exception as e:
        print(f"[Error] 无法读取 {large_img_path}: {e}")
        return [], [], ann_id_start, image_id_start

    if img.dtype == np.uint16:
        print("  -> 16bit 检测，执行 2% 拉伸到 8bit")
        img = linear_stretch(img)
    else:
        print("  -> 已是 8bit，跳过拉伸")

    h, w, _ = img.shape
    new_w = int(w * target_scale)
    new_h = int(h * target_scale)
    print(f"  -> 放大 {target_scale} 倍到 {new_w}x{new_h}")

    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    os.makedirs(output_dir, exist_ok=True)

    stride = crop_size - overlap
    if stride <= 0:
        raise ValueError(f"Stride must be positive, got crop_size={crop_size}, overlap={overlap}")
    rows = math.ceil((new_h - crop_size) / stride) + 1
    cols = math.ceil((new_w - crop_size) / stride) + 1

    img_resized_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

    new_images = []
    new_anns_all = []
    image_id = image_id_start
    ann_id = ann_id_start

    for r in range(rows):
        for c in range(cols):
            y_start = int(r * stride)
            x_start = int(c * stride)
            y_end = min(y_start + crop_size, new_h)
            x_end = min(x_start + crop_size, new_w)
            if y_end - y_start < crop_size:
                y_start = max(0, y_end - crop_size)
            if x_end - x_start < crop_size:
                x_start = max(0, x_end - crop_size)

            crop = img_resized_bgr[y_start:y_start + crop_size, x_start:x_start + crop_size]
            if crop.mean() < min_crop_mean:
                continue
            ext = ".jpg" if save_format == "jpg" else ".png"
            save_name = f"{base_name}_crop_{r}_{c}{ext}"
            cv2.imwrite(os.path.join(output_dir, save_name), crop)

            if anns_for_img is not None:
                new_anns, ann_id = process_annotations_for_crop(
                    anns_for_img,
                    scale=target_scale,
                    x0=x_start,
                    y0=y_start,
                    crop_size=crop_size,
                    ann_id_start=ann_id,
                    image_id=image_id,
                    min_object_area=min_object_area,
                )
                new_anns_all.extend(new_anns)

            new_images.append({
                "id": image_id,
                "file_name": save_name,
                "width": crop_size,
                "height": crop_size,
            })
            image_id += 1

    print(f"  -> 完成，生成 {len(new_images)} 张切片")
    return new_images, new_anns_all, ann_id, image_id


def main():
    parser = argparse.ArgumentParser(description="卫星大图预处理 + 切片，可选同步裁剪标注")
    parser.add_argument("--input_dir", default="/data/zfx/datasets/Jilin-1", help="大图目录")
    parser.add_argument("--output_dir", default="/data/zfx/datasets/Jilin-1/train", help="切片输出目录")
    parser.add_argument("--target_scale", type=float, default=2.5, help="放大倍数")
    parser.add_argument("--crop_size", type=int, default=320, help="切片尺寸")
    parser.add_argument("--overlap", type=int, default=100, help="切片重叠")
    parser.add_argument("--min_crop_mean", type=float, default=5.0, help="丢弃均值过低的黑片")
    parser.add_argument("--min_object_area", type=float, default=16.0, help="裁剪后最小保留目标面积 (像素)")
    parser.add_argument("--save_format", choices=["jpg", "png"], default="jpg", help="输出切片格式")
    parser.add_argument("--ann", type=str, default=None, help="可选：COCO 标注 json，若提供则同步裁剪标注")
    parser.add_argument("--out_ann", type=str, default=None, help="输出标注 json 路径，默认 output_dir/annotation_cropped.json")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    anns_index = None
    categories = []
    info = {}
    licenses = []
    if args.ann:
        anns_index, categories, info, licenses = load_ann_index(Path(args.ann))
        print(f"Loaded annotations from {args.ann}")

    tif_files = glob.glob(str(input_dir / "*.TIF")) + glob.glob(str(input_dir / "*.tif"))
    print(f"找到 {len(tif_files)} 个大图文件。")

    all_images = []
    all_annotations = []
    ann_id = 1
    image_id = 1

    for tif_file in tif_files:
        fname = Path(tif_file).name
        anns_for_img = None
        if anns_index is not None:
            anns_for_img = anns_index.get(fname)
            if anns_for_img is None:
                anns_for_img = anns_index.get(fname.lower()) or anns_index.get(fname.upper())
        imgs, anns, ann_id, image_id = preprocess_image(
            tif_file,
            output_dir=str(output_dir),
            target_scale=args.target_scale,
            crop_size=args.crop_size,
            overlap=args.overlap,
            anns_for_img=anns_for_img,
            ann_id_start=ann_id,
            image_id_start=image_id,
            min_crop_mean=args.min_crop_mean,
            min_object_area=args.min_object_area,
            save_format=args.save_format,
        )
        all_images.extend(imgs)
        all_annotations.extend(anns)

    if args.ann:
        out_ann_path = Path(args.out_ann) if args.out_ann else (output_dir / "annotation_cropped.json")
        new_ann = {
            "images": all_images,
            "annotations": all_annotations,
            "categories": categories,
            "info": info,
            "licenses": licenses,
        }
        with out_ann_path.open("w") as f:
            json.dump(new_ann, f)
        print(f"写出裁剪后标注到 {out_ann_path}")


if __name__ == "__main__":
    """
    用于将目标文件夹中的图像切片为指定大小的补丁，并可选同步裁剪 COCO 格式的标注。
    python util/preprocess_crop.py \
        --input_dir /data/zfx/datasets/WHU/train \
        --output_dir /data/zfx/datasets/WHUuda2/train \
        --target_scale 1 --crop_size 320 --overlap 100 \
        --min_crop_mean 5 --min_object_area 16 --save_format jpg \
        --ann /data/zfx/datasets/WHU/annotation/train.json \
        --out_ann /data/zfx/datasets/WHUuda2/train/annotation_cropped.json
    """
    main()