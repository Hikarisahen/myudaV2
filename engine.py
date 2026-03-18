# ------------------------------------------------------------------------
# Deformable DETR (MRT Optimized Version)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py (Modified for UDA with MRT)
"""
import math
import os
import sys
from typing import Iterable
import copy
import gc 

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.ops as ops # [新增] 用于 Box NMS
import util.misc as utils
from util.box_ops import box_cxcywh_to_xyxy
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from preprocess_annotation import uniform_sampling_vectorized

# =============================================================================
# Helper Class 1: MRT Dynamic Threshold (Source-Guided)
# =============================================================================
class DynamicThreshold:
    """
    基于 MRT 论文附录公式 (1) 的动态阈值控制器。
    Threshold = gamma * Old + (1-gamma) * a * (Mean_Source_Conf)^b
    
    """
    def __init__(self, num_classes, 
                 min_threshold=0.18, # 放宽下限，让早期目标域不过度空缺
                 initial_threshold=0.28, # 更低初始值，便于起步
                 max_threshold=0.45, # 论文上限 
                 gamma=0.95, 
                 a=0.8, # 论文参数 a [cite: 508]
                 b=1.0  # 论文参数 b [cite: 508]
                 ):
        self.thresholds = torch.full((num_classes,), initial_threshold).cuda()
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.gamma = gamma
        self.a = a
        self.b = b

    def update(self, source_logits, source_labels):
        """
        利用 Source 域的预测置信度更新阈值。
        关键点：只统计预测正确的、高置信度的 Source 样本。
        """
        with torch.no_grad():
            probs = source_logits.sigmoid()
            scores, pred_labels = probs.max(dim=-1) # (B, Q)
            
            # 展平以便统计
            scores_flat = scores.view(-1)
            pred_labels_flat = pred_labels.view(-1)
            
            # 获取当前 batch 存在的 GT 类别
            present_classes = torch.cat(source_labels).unique()
            
            for c in present_classes:
                c = c.long().item()
                if c >= len(self.thresholds): continue

                # 1. 找到预测为该类别的所有 Queries
                cls_mask = (pred_labels_flat == c)
                
                if cls_mask.sum() > 0:
                    cls_scores = scores_flat[cls_mask]
                    
                    # 2. 更加鲁棒的统计：只取 Top 50% 的分数，排除掉背景误检的低分
                    # MRT 论文思想：source confidence scores of positive instances [cite: 506]
                    k = max(1, int(cls_scores.numel() * 0.5))
                    top_scores, _ = torch.topk(cls_scores, k)
                    avg_score = top_scores.mean().item()
                    
                    # 3. MRT 公式更新 
                    # term = a * (avg_score ^ b)
                    term = self.a * (avg_score ** self.b)
                    
                    current_thr = self.thresholds[c].item()
                    new_thr = self.gamma * current_thr + (1 - self.gamma) * term
                    
                    # 4. 硬截断保护 (Safety Clamp) 
                    new_thr = max(new_thr, self.min_threshold)
                    new_thr = min(new_thr, self.max_threshold)
                    
                    self.thresholds[c] = new_thr

    def get_threshold(self, category_id):
        if category_id >= len(self.thresholds): return 0.4 # Default fallback
        return self.thresholds[category_id]

# =============================================================================
# Helper Function 2: FDA (Fourier Domain Adaptation)
# =============================================================================
def FDA_source_to_target(src_img, tgt_img, beta=0.01):
    """
    内存优化的 FDA：原地操作尽可能减少中间变量。
    """
    with torch.no_grad():
        # rfft2 节省一半内存
        fft_src = torch.fft.rfft2(src_img.clone(), dim=(-2, -1))
        fft_tgt = torch.fft.rfft2(tgt_img, dim=(-2, -1)) # tgt_img 只读，不clone节省内存

        amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)
        amp_tgt = torch.abs(fft_tgt) # pha_tgt 不需要，省内存

        _, _, h, w = src_img.shape
        b_h = int(h * beta)
        b_w = int(w * beta)
        
        # 交换低频幅度
        amp_src[..., :b_h, :b_w] = amp_tgt[..., :b_h, :b_w]
        amp_src[..., -b_h:, :b_w] = amp_tgt[..., -b_h:, :b_w]
        
        # 还原
        fft_src_mutated = torch.polar(amp_src, pha_src)
        src_in_tgt_style = torch.fft.irfft2(fft_src_mutated, s=(h, w), dim=(-2, -1))

        return src_in_tgt_style

# =============================================================================
# Helper Function 4: Corner Filtering & Utils
# =============================================================================
def get_poly_cosines(poly: np.ndarray) -> np.ndarray:
    diff_prev = poly - np.roll(poly, 1, axis=0)
    diff_next = np.roll(poly, -1, axis=0) - poly
    norm_prev = np.linalg.norm(diff_prev, axis=1, keepdims=True) + 1e-6
    norm_next = np.linalg.norm(diff_next, axis=1, keepdims=True) + 1e-6
    vec_prev = diff_prev / norm_prev
    vec_next = diff_next / norm_next
    return np.sum(vec_prev * vec_next, axis=1)

def apply_corner_nms(poly: np.ndarray, corner_scores: np.ndarray, corner_thresh: float, dist_thresh: float) -> np.ndarray:
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
    valid_rows = (~mask).any(dim=1)
    valid_cols = (~mask).any(dim=0)
    h = int(valid_rows.sum().item())
    w = int(valid_cols.sum().item())
    return h, w

def process_polygon_with_corners(poly_norm: torch.Tensor, corner_scores: torch.Tensor, img_h: int, img_w: int,
                                 corner_thresh: float, nms_thresh: float, enable_nms: bool, num_corners: int = 64):
    if img_h <= 0 or img_w <= 0: return None, None
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
        if corner_mask.sum() < 3: # Fallback
            corner_mask = scores_np > corner_thresh

    if corner_mask.sum() < 3:
        corner_mask = np.ones(poly_np.shape[0], dtype=bool)

    poly_for_sampling = poly_np[corner_mask]
    sampled_flat, corner_label = uniform_sampling_vectorized(poly_for_sampling.flatten(), num_corners)
    if sampled_flat is None: return None, None

    sampled_pts = torch.as_tensor(sampled_flat, dtype=torch.float32, device=poly_norm.device).reshape(num_corners, 2)
    sampled_pts[:, 0] /= float(img_w)
    sampled_pts[:, 1] /= float(img_h)
    sampled_corner_label = torch.as_tensor(corner_label, dtype=torch.float32, device=poly_norm.device)
    return sampled_pts, sampled_corner_label

# =============================================================================
# Helper Function 3: EMA Update
# =============================================================================
@torch.no_grad()
def update_teacher(student_model, teacher_model, alpha=0.998):
    """
    Teacher = alpha * Teacher + (1-alpha) * Student
    调低 alpha 让 Teacher 更快跟随，早期目标域更易出框
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
                    args=None):
    
    student.train()
    teacher.eval()
    criterion.train()
    criterion.epoch = epoch

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_sup', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_unsup', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mae', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = 'Epoch: [{}] (UDA)'.format(epoch)
    print_freq = 50

    num_classes = criterion.num_classes
    if not hasattr(criterion, 'dynamic_threshold'):
        criterion.dynamic_threshold = DynamicThreshold(num_classes)

    prefetcher_src = data_prefetcher(data_loader_source, device, prefetch=True)
    prefetcher_tgt = data_prefetcher(data_loader_target, device, prefetch=True)
    
    src_samples, src_targets = prefetcher_src.next()
    tgt_samples, _ = prefetcher_tgt.next()

    iter_per_epoch = min(len(data_loader_source), len(data_loader_target))
    pseudo_conf_sum = 0.0
    pseudo_conf_count = 0
    pseudo_conf_kept_sum = 0.0
    pseudo_conf_kept_count = 0
    pseudo_thr_raw_sum = 0.0
    pseudo_thr_effective_sum = 0.0
    pseudo_thr_target_ema_sum = 0.0
    pseudo_thr_quantile_sum = 0.0
    pseudo_thr_count = 0

    thr_quantile = float(getattr(args, 'pseudo_thr_quantile', 0.94))
    thr_quantile = min(0.99, max(0.5, thr_quantile))
    thr_ema_momentum = float(getattr(args, 'pseudo_thr_target_ema_momentum', 0.95))
    thr_ema_momentum = min(0.999, max(0.0, thr_ema_momentum))
    thr_min = float(getattr(args, 'pseudo_thr_min', 0.30))
    thr_max = float(getattr(args, 'pseudo_thr_max', 0.60))
    if thr_max < thr_min:
        thr_max = thr_min
    source_w = float(getattr(args, 'pseudo_thr_source_weight', 0.20))
    target_w = float(getattr(args, 'pseudo_thr_target_weight', 0.80))
    w_sum = source_w + target_w
    if w_sum <= 1e-12:
        source_w, target_w = 0.5, 0.5
        w_sum = 1.0
    source_w /= w_sum
    target_w /= w_sum

    if not hasattr(criterion, 'pseudo_target_thr_ema'):
        init_thr = float(getattr(args, 'pseudo_thr_init', 0.34))
        criterion.pseudo_target_thr_ema = min(thr_max, max(thr_min, init_thr))
    
    # === [MRT Strategy] Burn-in Period ===
    # 前若干 epoch 只做 Source + MAE，完全忽略 Teacher 伪标签
    burn_in_epochs = getattr(args, 'burn_in_epochs', 5)
    is_burn_in = epoch < burn_in_epochs
    
    for _ in metric_logger.log_every(range(iter_per_epoch), print_freq, header):
        if src_samples is None or tgt_samples is None:
            break

        # 1. 计算 Lambda (Unsup 权重)
        if is_burn_in:
            lambda_unsup = 0.0
        else:
            # Burn-in 后，缓慢增加权重 (Ramp-up)
            # 例如 10-20 epoch 从 0 线性增加到 target_lambda
            ramp_epochs = 80
            progress = min(1.0, max(0.0, epoch - burn_in_epochs) / ramp_epochs)
            lambda_unsup = args.lambda_unsup * progress
            
        # MAE 权重随时间衰减 (参考 MRT Paper Table 5)
        lambda_mae = args.lambda_mae * (1.0 - epoch / args.epochs)

        src_imgs = src_samples.tensors
        tgt_imgs = tgt_samples.tensors

        # 2. FDA Style Transfer (Source -> Target Style)
        # 即使在 burn-in 阶段也做 FDA，让 Student 适应 Target 风格
        src_imgs_stylized = FDA_source_to_target(src_imgs, tgt_imgs, beta=0.05)
        src_samples_stylized = utils.NestedTensor(src_imgs_stylized, src_samples.mask)

        # 3. Teacher Generate Pseudo Labels
        # [优化] Burn-in 阶段跳过此步骤，省显存
        pseudo_targets = []
        if not is_burn_in and lambda_unsup > 0:
            corner_thresh = getattr(args, 'pseudo_corner_thresh', 0.45)
            corner_nms_thresh = getattr(args, 'pseudo_corner_nms_thresh', 10.0)
            enable_corner_nms = not getattr(args, 'disable_pseudo_corner_nms', False)
            pseudo_min_box_area = getattr(args, 'pseudo_min_box_area', 0.004)
            pseudo_topk = getattr(args, 'pseudo_topk', 25)
            
            with torch.no_grad():
                teacher_output = teacher(tgt_samples)
                teacher_polys = teacher_output.get('pred_polys_evolve_1', teacher_output['pred_polys_init'])
                teacher_logits = teacher_output['pred_logits']
                teacher_boxes = teacher_output['pred_boxes']
                
                if 'pred_vtx_logits_evolve_1' in teacher_output:
                    teacher_vtx = teacher_output['pred_vtx_logits_evolve_1']
                elif 'pred_corners_init' in teacher_output:
                    teacher_vtx = teacher_output['pred_corners_init']
                else:
                    teacher_vtx = None

                probs = teacher_logits.sigmoid()
                top_scores, top_labels = probs.max(dim=-1)
                pseudo_conf_sum += top_scores.sum().item()
                pseudo_conf_count += top_scores.numel()

                if top_scores.numel() > 0:
                    q_thr = torch.quantile(top_scores.reshape(-1), thr_quantile).item()
                else:
                    q_thr = criterion.pseudo_target_thr_ema
                target_ema = thr_ema_momentum * criterion.pseudo_target_thr_ema + (1.0 - thr_ema_momentum) * q_thr
                target_ema = min(thr_max, max(thr_min, target_ema))
                criterion.pseudo_target_thr_ema = target_ema

                raw_thr = float(criterion.dynamic_threshold.get_threshold(0))
                fused_thr = source_w * raw_thr + target_w * target_ema
                effective_thr = min(thr_max, max(thr_min, fused_thr))

                for i in range(len(top_scores)):
                    thr = effective_thr
                    pseudo_thr_raw_sum += raw_thr
                    pseudo_thr_effective_sum += effective_thr
                    pseudo_thr_target_ema_sum += target_ema
                    pseudo_thr_quantile_sum += q_thr
                    pseudo_thr_count += 1
                    
                    # 3.1 初步筛选
                    keep_idxs = torch.where(top_scores[i] > thr)[0]
                    if keep_idxs.numel() > 0:
                        kept_scores = top_scores[i][keep_idxs]
                        pseudo_conf_kept_sum += kept_scores.sum().item()
                        pseudo_conf_kept_count += kept_scores.numel()
                    
                    if len(keep_idxs) == 0:
                        # Append empty
                        pseudo_targets.append({
                            'labels': torch.tensor([], dtype=torch.long, device=device),
                            'boxes': torch.empty((0, 4), device=device),
                            'poly_coords': torch.empty((0, 64, 2), device=device),
                            'corner_labels': torch.empty((0, 64), device=device)
                        })
                        continue
                    
                    # 提取数据
                    sel_boxes = teacher_boxes[i][keep_idxs]
                    sel_scores = top_scores[i][keep_idxs]
                    sel_polys = teacher_polys[i][keep_idxs]
                    sel_labels = top_labels[i][keep_idxs]
                    sel_vtx = teacher_vtx[i][keep_idxs] if teacher_vtx is not None else None
                    
                    # 3.2 [新增] Box NMS 过滤重叠框
                    # DETR cxcywh -> xyxy
                    sel_boxes_xyxy = box_cxcywh_to_xyxy(sel_boxes)
                    keep_nms = ops.nms(sel_boxes_xyxy, sel_scores, iou_threshold=0.4)
                    
                    # 应用 NMS
                    sel_boxes = sel_boxes[keep_nms]
                    sel_scores = sel_scores[keep_nms]
                    sel_polys = sel_polys[keep_nms]
                    sel_labels = sel_labels[keep_nms]
                    sel_vtx = sel_vtx[keep_nms] if sel_vtx is not None else None

                    if sel_boxes.shape[0] == 0:
                        pseudo_targets.append({
                            'labels': torch.tensor([], dtype=torch.long, device=device),
                            'boxes': torch.empty((0, 4), device=device),
                            'poly_coords': torch.empty((0, 64, 2), device=device),
                            'corner_labels': torch.empty((0, 64), device=device)
                        })
                        continue

                    # 3.2.1 [新增] 过滤过小伪框，抑制目标域小伪影
                    box_area = sel_boxes[:, 2].clamp(min=0) * sel_boxes[:, 3].clamp(min=0)
                    keep_area = torch.where(box_area >= pseudo_min_box_area)[0]

                    if keep_area.shape[0] == 0:
                        pseudo_targets.append({
                            'labels': torch.tensor([], dtype=torch.long, device=device),
                            'boxes': torch.empty((0, 4), device=device),
                            'poly_coords': torch.empty((0, 64, 2), device=device),
                            'corner_labels': torch.empty((0, 64), device=device)
                        })
                        continue

                    sel_boxes = sel_boxes[keep_area]
                    sel_scores = sel_scores[keep_area]
                    sel_polys = sel_polys[keep_area]
                    sel_labels = sel_labels[keep_area]
                    sel_vtx = sel_vtx[keep_area] if sel_vtx is not None else None

                    # 3.2.2 [新增] 每图仅保留高分 Top-K，抑制密集噪声伪框
                    if pseudo_topk > 0 and sel_scores.shape[0] > pseudo_topk:
                        topk_idx = torch.topk(sel_scores, k=pseudo_topk).indices
                        sel_boxes = sel_boxes[topk_idx]
                        sel_scores = sel_scores[topk_idx]
                        sel_polys = sel_polys[topk_idx]
                        sel_labels = sel_labels[topk_idx]
                        sel_vtx = sel_vtx[topk_idx] if sel_vtx is not None else None
                    
                    # 3.3 处理多边形和角点 (含 Corner NMS)
                    img_h, img_w = _get_valid_hw_from_mask(tgt_samples.mask[i])
                    
                    processed_polys = []
                    processed_corner_labels = []
                    processed_boxes = []
                    processed_labels = []
                    
                    for k_idx in range(sel_polys.shape[0]):
                        c_scores = None if sel_vtx is None else sel_vtx[k_idx].sigmoid()
                        
                        sampled_poly, sampled_corner = process_polygon_with_corners(
                            sel_polys[k_idx], c_scores, img_h, img_w,
                            corner_thresh, corner_nms_thresh, enable_corner_nms, num_corners=64
                        )
                        
                        if sampled_poly is not None:
                            processed_polys.append(sampled_poly)
                            processed_corner_labels.append(sampled_corner)
                            processed_boxes.append(sel_boxes[k_idx])
                            processed_labels.append(sel_labels[k_idx])
                    
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
        
        # 4. Student Forward & Loss Calculation
        import contextlib
        def get_no_sync_context(m):
            return m.no_sync() if hasattr(m, "no_sync") else contextlib.nullcontext()

        weight_dict = criterion.weight_dict
        optimizer.zero_grad()
        
        # 4.1 Source Loss (Always Active)
        # 只有当后面还有 Unsup 或 MAE 步骤时，才用 no_sync
        has_next_pass = (not is_burn_in and lambda_unsup > 0) or (lambda_mae > 0)
        ctx_sup = get_no_sync_context(student) if has_next_pass else contextlib.nullcontext()
        
        with ctx_sup:
            src_outputs = student(src_samples_stylized)
            loss_dict_sup = criterion(src_outputs, src_targets)
            losses_sup = sum(loss_dict_sup[k] * weight_dict[k] for k in loss_dict_sup.keys() if k in weight_dict)
            losses_sup.backward()
        
        # 4.2 Unsup Loss (Burn-in 期间跳过)
        losses_unsup = torch.tensor(0.0, device=device)
        loss_dict_unsup = {}
        
        has_mae_pass = (lambda_mae > 0)
        ctx_unsup = get_no_sync_context(student) if has_mae_pass else contextlib.nullcontext()

        if not is_burn_in and lambda_unsup > 0:
            with ctx_unsup:
                tgt_outputs = student(tgt_samples)
                loss_dict_unsup = criterion(tgt_outputs, pseudo_targets)
                unsup_keys = ['loss_ce', 'loss_bbox', 'loss_giou', 'loss_poly_consistency']
                losses_unsup = sum(loss_dict_unsup[k] * weight_dict[k] for k in loss_dict_unsup.keys() if k in weight_dict and k in unsup_keys)
                (losses_unsup * lambda_unsup).backward()

        # 4.3 MAE Loss
        losses_mae_weighted = torch.tensor(0.0, device=device)
        loss_mae = torch.tensor(0.0, device=device)
        
        if lambda_mae > 0:
            # Last pass needs sync
            mae_res = student(tgt_samples, mask_ratio=args.mask_ratio)
            if 'loss_mae' in mae_res:
                loss_mae = mae_res['loss_mae']
                losses_mae_weighted = loss_mae * lambda_mae
                losses_mae_weighted.backward()

        # 5. Logging & Optimizer Step
        total_loss_val = losses_sup.item() + lambda_unsup * losses_unsup.item() + losses_mae_weighted.item()

        if not math.isfinite(total_loss_val):
            print(f"Loss is {total_loss_val}, stopping training")
            sys.exit(1)

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm)
        optimizer.step()

        # 6. Update Teacher & Threshold
        # Burn-in 阶段冻结 Teacher，避免把伪标签噪声或未稳定表征注入 Teacher
        if not is_burn_in:
            update_teacher(student, teacher)
            # [MRT] 使用 Source 预测结果来更新阈值，而不是 Target
            criterion.dynamic_threshold.update(src_outputs['pred_logits'], [t['labels'] for t in src_targets])

        metric_logger.update(loss=total_loss_val)
        metric_logger.update(loss_sup=losses_sup.item())
        metric_logger.update(loss_unsup=losses_unsup.item())
        metric_logger.update(loss_mae=loss_mae.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        src_samples, src_targets = prefetcher_src.next()
        tgt_samples, _ = prefetcher_tgt.next()

    optimizer.zero_grad(set_to_none=True)

    # [Active Memory Optimization]
    del src_samples, src_targets, tgt_samples, src_imgs_stylized, src_samples_stylized
    if 'teacher_output' in locals(): del teacher_output
    if 'pseudo_targets' in locals(): del pseudo_targets
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['pseudo_conf_mean'] = (pseudo_conf_sum / pseudo_conf_count) if pseudo_conf_count > 0 else float('nan')
    stats['pseudo_conf_kept_mean'] = (pseudo_conf_kept_sum / pseudo_conf_kept_count) if pseudo_conf_kept_count > 0 else float('nan')
    stats['pseudo_thr_raw_mean'] = (pseudo_thr_raw_sum / pseudo_thr_count) if pseudo_thr_count > 0 else float('nan')
    stats['pseudo_thr_effective_mean'] = (pseudo_thr_effective_sum / pseudo_thr_count) if pseudo_thr_count > 0 else float('nan')
    stats['pseudo_thr_target_ema_mean'] = (pseudo_thr_target_ema_sum / pseudo_thr_count) if pseudo_thr_count > 0 else float('nan')
    stats['pseudo_thr_quantile_mean'] = (pseudo_thr_quantile_sum / pseudo_thr_count) if pseudo_thr_count > 0 else float('nan')
    return stats

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