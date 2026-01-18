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

import torch
import torch.nn.functional as F
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher

# =============================================================================
# Helper Class 1: Dynamic Threshold (MRT Paper Appendix Section 2)
# =============================================================================
class DynamicThreshold:
    """
    动态调整伪标签筛选阈值 。
    阈值 = gamma * 旧阈值 + (1-gamma) * (源域平均置信度)
    """
    def __init__(self, num_classes, initial_threshold=0.3, max_threshold=0.5, gamma=0.9):
        # 初始阈值设为 0.3，上限 0.5 (参考 MRT 附录 Table 2)
        self.thresholds = torch.full((num_classes,), initial_threshold).cuda()
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
# Helper Function 3: EMA Update
# =============================================================================
@torch.no_grad()
def update_teacher(student_model, teacher_model, alpha=0.999):
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
        # 前 5 epoch 不加 unsup loss，让 student 先学好 source
        lambda_unsup = args.lambda_unsup * min(1.0, max(0.0, epoch - 5) / 5.0)
        # MAE 权重随时间衰减
        lambda_mae = args.lambda_mae * (1.0 - epoch / args.epochs)

        src_imgs = src_samples.tensors
        tgt_imgs = tgt_samples.tensors

        # 1. FDA Style Transfer
        src_imgs_stylized = FDA_source_to_target(src_imgs, tgt_imgs, beta=0.09)
        src_samples_stylized = utils.NestedTensor(src_imgs_stylized, src_samples.mask)

        # 2. Teacher Generate Pseudo Labels
        pseudo_targets = []
        with torch.no_grad():
            teacher_output = teacher(tgt_samples)
            # 优先使用 Evolve_1 (精度最高)
            teacher_polys = teacher_output.get('pred_polys_evolve_1', teacher_output['pred_polys_init'])
            teacher_logits = teacher_output['pred_logits']
            teacher_boxes = teacher_output['pred_boxes']
            
            # 尝试获取角点 logits
            if 'pred_vtx_logits_evolve_1' in teacher_output:
                teacher_vtx = teacher_output['pred_vtx_logits_evolve_1']
            else:
                teacher_vtx = torch.zeros_like(teacher_polys[..., 0])

            probs = teacher_logits.sigmoid()
            top_scores, top_labels = probs.max(dim=-1)

            for i in range(len(top_scores)):
                # 获取动态阈值 (类别0是建筑)
                thr = criterion.dynamic_threshold.get_threshold(0)
                keep = top_scores[i] > thr
                
                # 安全检查：如果没有一个框符合阈值
                if keep.sum() == 0:
                    # 添加一个 dummy target 防止死循环或报错
                    # 也可以选择不添加，但在 list comprehension 中需要处理空的情况
                    pseudo_targets.append({
                        'labels': torch.tensor([], dtype=torch.long, device=device),
                        # [修改] 必须显式指定形状为 (0, 4)，否则 matcher 会报 1D 错误
                        'boxes': torch.empty((0, 4), device=device), 
                        'poly_coords': torch.empty((0, 64, 2), device=device),
                        'corner_labels': torch.empty((0, 64), device=device)
                    })
                    continue

                # 生成角点标签 (sigmoid > 0.5)
                valid_vtx = teacher_vtx[i][keep]
                pseudo_corner = (valid_vtx.sigmoid() > 0.5).float()

                pseudo_targets.append({
                    'labels': top_labels[i][keep],
                    'boxes': teacher_boxes[i][keep],
                    'poly_coords': teacher_polys[i][keep],
                    'corner_labels': pseudo_corner
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
    return stats, coco_evaluator