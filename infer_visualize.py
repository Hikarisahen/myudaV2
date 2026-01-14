import argparse
import os
from pathlib import Path
import glob
import cv2  # [新增] 引入 OpenCV 以加速绘图
import numpy as np
import torch
from PIL import Image

from main import get_args_parser
from models import build_model
from datasets.coco import make_coco_transforms
from util.misc import nested_tensor_from_tensor_list


def load_model(args, checkpoint_path):
    model, criterion, postprocessors = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # [Fix] 自动过滤形状不匹配的权重 (如 class_embed)
    state_dict = checkpoint['model']
    model_state_dict = model.state_dict()
    keys_to_remove = []
    for k, v in state_dict.items():
        if k in model_state_dict:
            if v.shape != model_state_dict[k].shape:
                print(f"Shape mismatch for {k}: checkpoint {v.shape} vs model {model_state_dict[k].shape}. Dropping from checkpoint.")
                keys_to_remove.append(k)
    for k in keys_to_remove:
        del state_dict[k]

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: missing keys when loading checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: unexpected keys when loading checkpoint: {unexpected_keys}")
    model.to(args.device)
    model.eval()
    return model


def collect_images(image_path=None, image_dir=None):
    paths = []
    if image_path:
        paths.append(image_path)
    if image_dir:
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.JPG", "*.TIF")
        for ext in exts:
            paths.extend(sorted(glob.glob(os.path.join(image_dir, ext))))
    return paths


def get_poly_cosines(poly):
    """
    计算每个顶点的余弦值 (用于判断是否锐角)
    poly: (N, 2)
    """
    diff_prev = poly - np.roll(poly, 1, axis=0)
    diff_next = np.roll(poly, -1, axis=0) - poly
    norm_prev = np.linalg.norm(diff_prev, axis=1, keepdims=True) + 1e-6
    norm_next = np.linalg.norm(diff_next, axis=1, keepdims=True) + 1e-6
    
    vec_prev = diff_prev / norm_prev
    vec_next = diff_next / norm_next
    
    return np.sum(vec_prev * vec_next, axis=1)


def apply_nms(poly, corner_scores, corner_thresh, dist_thresh):
    """
    对单张多边形的角点进行 NMS (支持动态抑制策略)
    poly: (N, 2) 绝对坐标
    corner_scores: (N,) 角点得分
    corner_thresh: 得分阈值
    dist_thresh: 距离阈值 (像素)
    """
    candidate_idxs = np.where(corner_scores > corner_thresh)[0]
    if len(candidate_idxs) == 0:
        return np.zeros_like(corner_scores, dtype=bool)

    # 1. 计算每个点的几何锐度
    # cos值越小越尖 (1:直线, 0:直角, -1:折返)
    # 设定 "锐角" 阈值 (例如 45度 -> cos ~ 0.7)
    cosines = get_poly_cosines(poly)
    sharp_mask = cosines < 0.75  # 简单的启发式阈值
    min_dist_for_short_edge = 3.0 # 如果两个点都是锐角，允许的最小距离 (3像素)

    # 按得分降序排列
    sorted_indices = candidate_idxs[np.argsort(corner_scores[candidate_idxs])[::-1]]
    
    keep_indices = []
    for idx in sorted_indices:
        curr_coord = poly[idx]
        if not keep_indices:
            keep_indices.append(idx)
            continue
        
        # 计算与已保留点的距离
        kept_idxs_arr = np.array(keep_indices)
        kept_coords = poly[kept_idxs_arr]
        dists = np.linalg.norm(kept_coords - curr_coord, axis=1)
        
        # --- [动态 NMS 核心逻辑] ---
        # 默认阈值
        thresholds = np.full_like(dists, dist_thresh)
        
        # 如果当前点是锐角，且被比较的点也是锐角 -> 这是一个短边结构 -> 使用极小阈值
        if sharp_mask[idx]:
            # 找到那些也是锐角的已保留点
            is_kept_sharp = sharp_mask[kept_idxs_arr]
            # 对这些点，将抑制阈值降低 (允许共存)
            thresholds[is_kept_sharp] = min_dist_for_short_edge
            
        # 只要与任何一个保留点的距离小于对应的阈值，就被抑制
        if np.all(dists > thresholds):
            keep_indices.append(idx)
            
    mask = np.zeros_like(corner_scores, dtype=bool)
    mask[keep_indices] = True
    return mask


def visualize_image(model, img_path, transform, device, save_dir, score_thresh=0.5, corner_thresh=0.5, only_corner=False, enable_nms=False, nms_thresh=10.0):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    img_t, _ = transform(img, None)
    samples = nested_tensor_from_tensor_list([img_t]).to(device)

    with torch.no_grad():
        outputs = model(samples)

    pred_logits = outputs["pred_logits"][0]
    # [Fix] 适配新模型输出键名 (优先取 evolve_1, 其次 init, 最后尝试旧键名)
    # 注意：outputs["pred_polys_xxx"] 的形状通常是 [Batch, Queries, N, 2]
    # 这里我们只推理了一张图，所以取 [0]
    pred_polys_batch = outputs.get("pred_polys_evolve_1", outputs.get("pred_polys_init", outputs.get("pred_polys")))
    pred_polys = pred_polys_batch[0] # [Q, N, 2]
    
    # [Fix] 适配新模型角点键名
    pred_corners_logits_batch = outputs.get("pred_vtx_logits_evolve_1", outputs.get("pred_corners_init", outputs.get("pred_corners")))
    pred_corners_logits = pred_corners_logits_batch[0] # [Q, N]
    pred_corners = pred_corners_logits.sigmoid()

    # [Fix] 针对单类别 (num_classes=1) 的处理
    # 如果 num_classes=1，pred_logits 形状为 [Q, 1] 或 [Q, 2] (取决于是否包含背景类)
    # 但在 Deformable DETR 中，通常是 [Q, num_classes] (sigmoid) 或 [Q, num_classes+1] (softmax)
    # 这里假设是 sigmoid 且 num_classes=1，则 pred_logits 形状为 [Q, 1]
    
    probs = pred_logits.sigmoid()
    if probs.shape[-1] == 1:
        # 单类别情况，直接取值
        scores = probs.squeeze(-1)
    else:
        # 多类别情况，取除背景外的最大值
        scores, _ = probs[..., :-1].max(-1)

    keep = scores > score_thresh
    polys = pred_polys[keep].cpu().numpy()
    corners = pred_corners[keep].cpu().numpy()

    polys[..., 0] *= w
    polys[..., 1] *= h

    # [优化] 使用 OpenCV 进行绘图，显著提升速度
    # Convert PIL to BGR for OpenCV
    img_np = np.array(img)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    overlay = img_cv2.copy()

    for poly, corner in zip(polys, corners):
        if enable_nms:
            is_corner = apply_nms(poly, corner, corner_thresh, nms_thresh)
        else:
            is_corner = corner > corner_thresh

        # 1. 确定多边形轮廓点
        if only_corner and is_corner.sum() >= 3:
            draw_poly = poly[is_corner]
        else:
            draw_poly = poly
        
        # 转换为 int32 用于 cv2 绘图
        pts = draw_poly.astype(np.int32).reshape((-1, 1, 2))

        # 2. 绘制半透明填充 (Cyan: B=255, G=255, R=0)
        cv2.fillPoly(overlay, [pts], (255, 255, 0))
        
        # 3. 绘制边框线条 (Cyan)
        cv2.polylines(img_cv2, [pts], True, (255, 255, 0), 1, cv2.LINE_AA)

        # 4. 绘制角点 (Red: B=0, G=0, R=255)
        if is_corner.any():
            for pt in poly[is_corner]:
                cv2.circle(img_cv2, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)

        # 5. [新增] 在非 only_corner 模式下，显式显示非角点 (Yellow: B=0, G=255, R=255)
        if not only_corner:
            not_corner = ~is_corner
            if not_corner.any():
                for pt in poly[not_corner]:
                    cv2.circle(img_cv2, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)

    # Alpha blend: 0.2 transparency
    alpha = 0.2
    cv2.addWeighted(overlay, alpha, img_cv2, 1 - alpha, 0, img_cv2)

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / (Path(img_path).stem + "_vis.png")
    cv2.imwrite(str(out_path), img_cv2)
    print(f"Saved: {out_path}")


def main():
    parser = get_args_parser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pth")
    parser.add_argument("--image_path", help="Single image for inference")
    parser.add_argument("--image_dir", help="Folder containing images for batch inference")
    # 避免与 get_args_parser 里的 --output_dir 冲突，改名为 vis_output_dir
    parser.add_argument("--vis_output_dir", default="./vis_out", help="Directory to save visualizations")
    parser.add_argument("--score_thresh", type=float, default=0.5, help="Object score threshold")
    parser.add_argument("--corner_thresh", type=float, default=0.5, help="Corner score threshold")
    parser.add_argument("--onlycorner", action="store_true", help="Only draw polygon formed by corner points")
    parser.add_argument("--enable_nms", action="store_true", help="Enable NMS for corner points")
    parser.add_argument("--nms_thresh", type=float, default=10.0, help="NMS distance threshold in pixels")

    args = parser.parse_args()

    if not args.image_path and not args.image_dir:
        raise ValueError("Please provide --image_path or --image_dir")

    args.device = torch.device(args.device)
    model = load_model(args, args.checkpoint)

    transform = make_coco_transforms("val")
    save_dir = Path(args.vis_output_dir)

    img_paths = collect_images(args.image_path, args.image_dir)
    if not img_paths:
        raise ValueError("No images found with given path(s)")

    for p in img_paths:
        visualize_image(
            model,
            p,
            transform,
            args.device,
            save_dir,
            args.score_thresh,
            args.corner_thresh,
            args.onlycorner,
            args.enable_nms,
            args.nms_thresh,
        )


if __name__ == "__main__":
    main()
    """
    推理可视化单张图像
    python infer_visualize.py \
        --checkpoint /path/to/checkpoint.pth \
        --image_path /home/zfx/datasets/WHU/validation/2000129.TIF \
        --vis_output_dir ./vis_out \
        --score_thresh 0.5 \
        --corner_thresh 0.5

    推理可视化批量图像
    python infer_visualize.py \
        --checkpoint "/home/data/zfx/DETR/SL1andfocal_train/checkpoint.pth" \
        --image_dir /home/zfx/datasets/WHU/test \
        --vis_output_dir /home/data/zfx/DETR/SL1andfocal_train/vis_out
    """
