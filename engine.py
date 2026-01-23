# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py (Modified for UDA)
"""
import math
import os
import sys
from typing import Iterable
import copy

import numpy as np
import torch
import torch.nn.functional as F
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from preprocess_annotation import uniform_sampling_vectorized

# =============================================================================
# Helper Class 1: Dynamic Threshold (MRT Paper Appendix Section 2)
# =============================================================================
class DynamicThreshold:
    """
    动态调整伪标签筛选阈值 。
    阈值 = gamma * 旧阈值 + (1-gamma) * (源域平均置信度)
    """
    def __init__(self, num_classes, min_threshold=0.3, initial_threshold=0.45, max_threshold=0.55, gamma=0.9):
        # 初始阈值设为 0.4，上限 0.5 (参考 MRT 附录 Table 2修改)
        self.thresholds = torch.full((num_classes,), initial_threshold).cuda()
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.gamma = gamma

    def update(self, source_logits, source_labels):
        """
        利用源域(CrowdAI)的高质量预测结果来校准阈值
        """
        with torch.no_grad():
            probs = source_logits.sigmoid()
            # 获取每个样本预测的最可信类别的分数
            scores, pred_labels = probs.max(dim=-1) # (B, Q)
            
            # 我们不知道哪个 Query 匹配哪个 GT，简单起见，统计所有预测正确类别的最大分数
            # 或者更严谨地：只统计 hungarian match 上的 pair。
            # 这里采用一种简单的 heuristic: 
            # 只要预测的类别存在于当前图片的 GT 标签集合中，且分数够高，就认为是可信的统计对象
            # 扁平化处理
            scores_flat = scores.view(-1)
            pred_labels_flat = pred_labels.view(-1)
            
            # 获取当前 batch 所有出现的 GT 类别
            present_classes = torch.cat(source_labels).unique()
            
            for c in present_classes:
                c = c.long().item()
                if c >= len(self.thresholds): continue

                # 找到预测为该类别的所有 queries
                mask = (pred_labels_flat == c)
                if mask.sum() > 0:
                    cls_scores = scores_flat[mask]
                    # 取前 50% 的分数作为参考，避免被背景噪声拉低
                    k = max(1, int(cls_scores.numel() * 0.5))
                    top_scores, _ = torch.topk(cls_scores, k)
                    # [Fix] 使用 .item() 转换为 Python float，彻底断开 Graph 连接，防止内存泄漏
                    avg_score = top_scores.mean().item()
                    
                    # EMA 更新 (纯 Scalar 运算)
                    current_thr = self.thresholds[c].item()
                    new_thr = self.gamma * current_thr + (1 - self.gamma) * avg_score
                    new_thr = max(new_thr, self.min_threshold)
                    new_thr = min(new_thr, self.max_threshold)
                    
                    # 原地赋值 (Scalar -> Tensor Element)
                    self.thresholds[c] = new_thr

    def get_threshold(self, category_id):
        # 简单的越界保护
        if category_id >= len(self.thresholds): return 0.3
        return self.thresholds[category_id]

# =============================================================================
# Helper Function 2: FDA (Fourier Domain Adaptation)
# =============================================================================
def FDA_source_to_target(src_img, tgt_img, beta=0.01):
    """
    将 src_img 的低频风格替换为 tgt_img 的低频风格。
    使用 rfft2 (实数 FFT) 替代 fft2，减少 50% 的显存占用和计算量。
    """
    with torch.no_grad():
        # [优化] 使用 rfft2 只计算非冗余频域信息 (实数输入，复数输出，但宽度减半)
        fft_src = torch.fft.rfft2(src_img.clone(), dim=(-2, -1))
        fft_tgt = torch.fft.rfft2(tgt_img.clone(), dim=(-2, -1))

        # 提取幅度(Amplitude)和相位(Phase)
        amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)
        amp_tgt, pha_tgt = torch.abs(fft_tgt), torch.angle(fft_tgt)

        # 替换低频中心区域
        _, _, h, w = src_img.shape
        # rfft2 输出的最后一维宽度约为 w/2 + 1
        # 低频分量位于 (0, 0) 附近
        
        b_h = int(h * beta)
        b_w = int(w * beta)
        
        # rfft2 的低频布局：
        # Y轴: [0, b_h] 是正频率低频, [-b_h, end] 是负频率低频
        # X轴: 只有 [0, b_w] 部分，因为负频率部分是共轭对称的，rfft2 不存储
        
        # 左上角 (低频正频率部分)
        amp_src[..., :b_h, :b_w] = amp_tgt[..., :b_h, :b_w]
        # 左下角 (低频负频率部分)
        amp_src[..., -b_h:, :b_w] = amp_tgt[..., -b_h:, :b_w]
        
        # 注意：rfft2 不需要处理右半部分，因为 X 轴只有一半

        # 还原回图像
        fft_src_mutated = torch.polar(amp_src, pha_src)
        # [优化] 使用 irfft2
        src_in_tgt_style = torch.fft.irfft2(fft_src_mutated, s=(h, w), dim=(-2, -1))

        return src_in_tgt_style


# =============================================================================
# Helper Function 4: Corner Filtering + Uniform Sampling (Teacher Pseudo Labels)
# =============================================================================
def get_poly_cosines(poly: np.ndarray) -> np.ndarray:
    """
    Compute cosine of each interior angle for a polygon (N, 2).
    Used to identify sharp corners during NMS.
    """
    diff_prev = poly - np.roll(poly, 1, axis=0)
    diff_next = np.roll(poly, -1, axis=0) - poly
    norm_prev = np.linalg.norm(diff_prev, axis=1, keepdims=True) + 1e-6
    norm_next = np.linalg.norm(diff_next, axis=1, keepdims=True) + 1e-6

    vec_prev = diff_prev / norm_prev
    vec_next = diff_next / norm_next

    return np.sum(vec_prev * vec_next, axis=1)


def apply_corner_nms(poly: np.ndarray, corner_scores: np.ndarray, corner_thresh: float, dist_thresh: float) -> np.ndarray:
    """
    Corner-level NMS identical to infer_visualize.py, but pure numpy for training loop.
    Returns a boolean mask of kept corners.
    """
    candidate_idxs = np.where(corner_scores > corner_thresh)[0]
    if len(candidate_idxs) == 0:
        return np.zeros_like(corner_scores, dtype=bool)

    cosines = get_poly_cosines(poly)
    sharp_mask = cosines < 0.75
    min_dist_for_short_edge = 3.0

    sorted_indices = candidate_idxs[np.argsort(corner_scores[candidate_idxs])[::-1]]

    keep_indices = []
    for idx in sorted_indices:
        curr_coord = poly[idx]
        if not keep_indices:
            keep_indices.append(idx)
            continue

        kept_idxs_arr = np.array(keep_indices)
        kept_coords = poly[kept_idxs_arr]
        dists = np.linalg.norm(kept_coords - curr_coord, axis=1)

        thresholds = np.full_like(dists, dist_thresh)
        if sharp_mask[idx]:
            is_kept_sharp = sharp_mask[kept_idxs_arr]
            thresholds[is_kept_sharp] = min_dist_for_short_edge

        if np.all(dists > thresholds):
            keep_indices.append(idx)

    mask = np.zeros_like(corner_scores, dtype=bool)
    mask[keep_indices] = True
    return mask


def _get_valid_hw_from_mask(mask: torch.Tensor):
    """Derive valid (h, w) from NestedTensor mask (True = padding)."""
    valid_rows = (~mask).any(dim=1)
    valid_cols = (~mask).any(dim=0)
    h = int(valid_rows.sum().item())
    w = int(valid_cols.sum().item())
    return h, w


def process_polygon_with_corners(poly_norm: torch.Tensor, corner_scores: torch.Tensor, img_h: int, img_w: int,
                                 corner_thresh: float, nms_thresh: float, enable_nms: bool, num_corners: int = 64):
    """
    1) 将归一化多边形转换到像素坐标
    2) 依据角点分数筛选/去重角点
    3) 使用 preprocess_annotation 中的均匀采样逻辑插值到固定点数
    返回归一化后的采样点与角点标签 (shape: [num_corners, 2], [num_corners]).
    """
    if img_h <= 0 or img_w <= 0:
        return None, None

    poly_np = poly_norm.detach().cpu().numpy().copy()
    poly_np[:, 0] *= img_w
    poly_np[:, 1] *= img_h

    scores_np = None if corner_scores is None else corner_scores.detach().cpu().numpy()

    if scores_np is None:
        corner_mask = np.ones(poly_np.shape[0], dtype=bool)
    else:
        if enable_nms:
            corner_mask = apply_corner_nms(poly_np, scores_np, corner_thresh, nms_thresh)
        else:
            corner_mask = scores_np > corner_thresh

        if corner_mask.sum() < 3:
            corner_mask = scores_np > corner_thresh

    if corner_mask.sum() < 3:
        corner_mask = np.ones(poly_np.shape[0], dtype=bool)

    poly_for_sampling = poly_np[corner_mask]
    sampled_flat, corner_label = uniform_sampling_vectorized(poly_for_sampling.flatten(), num_corners)
    if sampled_flat is None:
        return None, None

    sampled_pts = torch.as_tensor(sampled_flat, dtype=torch.float32, device=poly_norm.device).reshape(num_corners, 2)
    sampled_pts[:, 0] /= float(img_w)
    sampled_pts[:, 1] /= float(img_h)
    sampled_corner_label = torch.as_tensor(corner_label, dtype=torch.float32, device=poly_norm.device)

    return sampled_pts, sampled_corner_label

# =============================================================================
# Helper Function 3: EMA Update
# =============================================================================
@torch.no_grad()
def update_teacher(student_model, teacher_model, alpha=0.9996):
    """
    Mean Teacher 更新逻辑: Teacher = alpha * Teacher + (1-alpha) * Student
    [cite: 108, 109]
    """
    for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)

# =============================================================================
# Main Training Function (Modified for UDA)
# =============================================================================
def train_one_epoch(student: torch.nn.Module, teacher: torch.nn.Module, 
                    criterion: torch.nn.Module,
                    data_loader_source: Iterable, data_loader_target: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    args=None): # 需要传入 args 以获取 lambda 参数
    
    student.train()
    teacher.eval() # Teacher 永远是 eval 模式
    criterion.train()
    criterion.epoch = epoch  # 传递 epoch 给 Loss 计算课程权重

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_sup', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_unsup', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mae', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = 'Epoch: [{}] (UDA)'.format(epoch)
    print_freq = 50

    # 初始化动态阈值控制器 (假设只有1类建筑，如果有更多类需调整)
    num_classes = criterion.num_classes
    if not hasattr(criterion, 'dynamic_threshold'):
        criterion.dynamic_threshold = DynamicThreshold(num_classes)

    prefetcher_src = data_prefetcher(data_loader_source, device, prefetch=True)
    prefetcher_tgt = data_prefetcher(data_loader_target, device, prefetch=True)
    
    src_samples, src_targets = prefetcher_src.next()
    tgt_samples, _ = prefetcher_tgt.next()

    # 简单的 zip 截断策略
    iter_per_epoch = min(len(data_loader_source), len(data_loader_target))
    
    for _ in metric_logger.log_every(range(iter_per_epoch), print_freq, header):
        if src_samples is None or tgt_samples is None:
            break

        # === 动态权重 ===
        # 前 10 epoch 不加 unsup loss，让 student 先学好 source
        burn_in_epochs = 10
        lambda_unsup = args.lambda_unsup * min(1.0, max(0.0, epoch - burn_in_epochs) / 5.0)
        # MAE 权重随时间衰减
        lambda_mae = args.lambda_mae * (1.0 - epoch / args.epochs)

        src_imgs = src_samples.tensors
        tgt_imgs = tgt_samples.tensors

        # 1. FDA Style Transfer
        src_imgs_stylized = FDA_source_to_target(src_imgs, tgt_imgs, beta=0.09)
        src_samples_stylized = utils.NestedTensor(src_imgs_stylized, src_samples.mask)

        # 2. Teacher Generate Pseudo Labels (corner processing + 64-point interpolation)
        pseudo_targets = []
        corner_thresh = getattr(args, 'pseudo_corner_thresh', 0.45)
        corner_nms_thresh = getattr(args, 'pseudo_corner_nms_thresh', 10.0)
        enable_corner_nms = not getattr(args, 'disable_pseudo_corner_nms', False)
        with torch.no_grad():
            teacher_output = teacher(tgt_samples)
            # 优先使用 Evolve_1 (精度最高)
            teacher_polys = teacher_output.get('pred_polys_evolve_1', teacher_output['pred_polys_init'])
            teacher_logits = teacher_output['pred_logits']
            teacher_boxes = teacher_output['pred_boxes']

            # 尝试获取角点 logits
            if 'pred_vtx_logits_evolve_1' in teacher_output:
                teacher_vtx = teacher_output['pred_vtx_logits_evolve_1']
            elif 'pred_corners_init' in teacher_output:
                teacher_vtx = teacher_output['pred_corners_init']
            else:
                teacher_vtx = None

            probs = teacher_logits.sigmoid()
            top_scores, top_labels = probs.max(dim=-1)

            for i in range(len(top_scores)):
                thr = criterion.dynamic_threshold.get_threshold(0)
                keep = top_scores[i] > thr

                img_h, img_w = _get_valid_hw_from_mask(tgt_samples.mask[i])

                if keep.sum() == 0:
                    pseudo_targets.append({
                        'labels': torch.tensor([], dtype=torch.long, device=device),
                        'boxes': torch.empty((0, 4), device=device),
                        'poly_coords': torch.empty((0, 64, 2), device=device),
                        'corner_labels': torch.empty((0, 64), device=device)
                    })
                    continue

                kept_polys = teacher_polys[i][keep]
                kept_boxes = teacher_boxes[i][keep]
                kept_labels = top_labels[i][keep]
                kept_corners = teacher_vtx[i][keep] if teacher_vtx is not None else None

                processed_polys = []
                processed_corner_labels = []
                processed_boxes = []
                processed_labels = []

                for det_idx in range(kept_polys.shape[0]):
                    corner_scores = None if kept_corners is None else kept_corners[det_idx].sigmoid()
                    sampled_poly, sampled_corner = process_polygon_with_corners(
                        kept_polys[det_idx],
                        corner_scores,
                        img_h,
                        img_w,
                        corner_thresh,
                        corner_nms_thresh,
                        enable_corner_nms,
                        num_corners=64,
                    )

                    if sampled_poly is None:
                        continue

                    processed_polys.append(sampled_poly)
                    processed_corner_labels.append(sampled_corner)
                    processed_boxes.append(kept_boxes[det_idx])
                    processed_labels.append(kept_labels[det_idx])

                if len(processed_polys) == 0:
                    pseudo_targets.append({
                        'labels': torch.tensor([], dtype=torch.long, device=device),
                        'boxes': torch.empty((0, 4), device=device),
                        'poly_coords': torch.empty((0, 64, 2), device=device),
                        'corner_labels': torch.empty((0, 64), device=device)
                    })
                    continue

                pseudo_targets.append({
                    'labels': torch.stack(processed_labels),
                    'boxes': torch.stack(processed_boxes),
                    'poly_coords': torch.stack(processed_polys),
                    'corner_labels': torch.stack(processed_corner_labels)
                })

        # 3. Student Forward
        import contextlib
        def get_no_sync_context(m):
            return m.no_sync() if hasattr(m, "no_sync") else contextlib.nullcontext()

        weight_dict = criterion.weight_dict

        optimizer.zero_grad()
        
        # 3.1 Source (Labeled + Stylized) -> Supervised Loss
        # 使用 no_sync 避免多次 DDP sync (除非这是唯一的 pass)
        # 如果既没有 Unsup 也没有 MAE，这里应该 sync。但一般 UDA 都有 Unsup/MAE。
        # 为了安全，我们只在后续有 pass 时用 no_sync。
        has_next_pass = (lambda_unsup > 0) or (lambda_mae > 0)
        ctx_sup = get_no_sync_context(student) if has_next_pass else contextlib.nullcontext()
        
        with ctx_sup:
            src_outputs = student(src_samples_stylized)
            loss_dict_sup = criterion(src_outputs, src_targets)
            losses_sup = sum(loss_dict_sup[k] * weight_dict[k] for k in loss_dict_sup.keys() if k in weight_dict)
            losses_sup.backward()
        
        # 3.2 Target (Unlabeled) -> Unsupervised Loss
        loss_dict_unsup = {}
        losses_unsup = torch.tensor(0.0, device=device)
        
        has_mae_pass = (lambda_mae > 0)
        ctx_unsup = get_no_sync_context(student) if has_mae_pass else contextlib.nullcontext()
        
        if lambda_unsup > 0:
            with ctx_unsup:
                tgt_outputs = student(tgt_samples)
                loss_dict_unsup = criterion(tgt_outputs, pseudo_targets)
                unsup_keys = ['loss_ce', 'loss_bbox', 'loss_giou', 'loss_poly_consistency']
                losses_unsup = sum(loss_dict_unsup[k] * weight_dict[k] for k in loss_dict_unsup.keys() if k in weight_dict and k in unsup_keys)
                (losses_unsup * lambda_unsup).backward()
        else:
            # 如果不跑 Unsup，需要确保 src_outputs 还在 (后续用到)，但不需要 tgt_outputs
            pass
        
        # 3.3 Target -> MAE Loss
        losses_mae_weighted = torch.tensor(0.0, device=device)
        loss_mae = torch.tensor(0.0, device=device)
        
        if lambda_mae > 0:
            # 这是最后一步，必须 Sync
            mae_res = student(tgt_samples, mask_ratio=args.mask_ratio)
            if 'loss_mae' in mae_res:
                loss_mae = mae_res['loss_mae']
                losses_mae_weighted = loss_mae * lambda_mae
                losses_mae_weighted.backward()

        # 4. Loss Aggregation (For Logging Only)
        # 注意: backward 已经分开执行了
        # [Fix] 使用 detached tensors 计算 total_loss 用于日志，避免持有整个 Graph
        total_loss_val = losses_sup.item() + lambda_unsup * losses_unsup.item() + losses_mae_weighted.item()

        if not math.isfinite(total_loss_val):
            print(f"Loss is {total_loss_val}, stopping training")
            print(f"Sup: {losses_sup.item()}, Unsup: {losses_unsup.item()}, MAE: {loss_mae.item()}")
            sys.exit(1)

        # optimizer.zero_grad() # 已经在最开始做了
        # total_loss.backward() # 已经在上面分步做了
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm)
        optimizer.step()

        # 5. Updates
        update_teacher(student, teacher)
        criterion.dynamic_threshold.update(src_outputs['pred_logits'], [t['labels'] for t in src_targets])

        # Logging
        metric_logger.update(loss=total_loss_val)
        metric_logger.update(loss_sup=losses_sup.item())
        metric_logger.update(loss_unsup=losses_unsup.item())
        metric_logger.update(loss_mae=loss_mae.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        src_samples, src_targets = prefetcher_src.next()
        tgt_samples, _ = prefetcher_tgt.next()

    # [Fix] Epoch 结束时清理梯度，释放显存 (set_to_none=True 更高效)
    optimizer.zero_grad(set_to_none=True)

    # [Active Memory Optimization]
    import gc
    del src_samples, src_targets, tgt_samples, loss_dict_sup, loss_dict_unsup, mae_res
    if 'total_loss_val' in locals(): del total_loss_val
    if 'teacher_output' in locals(): del teacher_output
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    # Evaluate 逻辑保持不变，通常只用 Teacher 模型进行测试
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    
    # [Memory Optimization] Clean up evaluator to free memory
    import gc
    del coco_evaluator
    del panoptic_evaluator
    del loss_dict, loss_dict_reduced, loss_dict_reduced_scaled, loss_dict_reduced_unscaled
    gc.collect()

    return stats, None # Do not return the evaluator object to avoid keeping references