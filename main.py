# ------------------------------------------------------------------------
# Deformable DETR (UDA Version: FDA + Mean Teacher + MAE)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import sys
import os
import copy  # [新增] 用于复制 Teacher 模型
import shlex

# === [新增] 强行加入算子目录到系统路径 ===
sys.path.append(os.path.join(os.path.dirname(__file__), 'models/ops'))
# ========================================

from torch.utils.tensorboard import SummaryWriter
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

import torch.multiprocessing
# === [新增] 强制使用文件系统策略，解决 Bad file descriptor/Resize 错误 ===
torch.multiprocessing.set_sharing_strategy('file_system')
# ===================================================================

import datasets
import util.misc as utils
from util.misc import nested_tensor_from_tensor_list
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch  # 这里调用的是修改后的支持 UDA 的 train_one_epoch
from models import build_model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 固定可视化使用的图片路径
CUSTOM_VIS_IMAGE_PATHS = [
    # # 为吉林一号（目标域）的图片路径，用于观察适应效果
    # "/data/zfx/datasets/CrowdAI/test_images/000000000041.jpg",
    # "/data/zfx/datasets/Jilin-1/train/images/tile_0_crop_49_64.jpg",
    # "/data/zfx/datasets/Jilin-1/train/images/tile_0_crop_68_82.jpg",
    # "/data/zfx/datasets/Jilin-1/train/images/tile_0_crop_79_5.jpg",
    # WHU
    # "/data/zfx/datasets/CrowdAI/test_images/000000000041.jpg",
    # "/data/zfx/datasets/WHUuda/train/images/1000121_crop_4_4.jpg",  # 建筑物
    # "/data/zfx/datasets/WHUuda/train/images/1000267_crop_0_2.jpg",  # 道路负样本
    # "/data/zfx/datasets/WHUuda/train/images/10002218_crop_4_3.jpg",  # 复杂建筑物
    # GoogleMap
    # "/data/zfx/datasets/CrowdAI/test_images/000000000041.jpg",
    # "/data/zfx/datasets/GoogleMap/train/images/target_domain_000121.jpg",  # 建筑物
    # "/data/zfx/datasets/GoogleMap/train/images/target_domain_000152.jpg",  
    # "/data/zfx/datasets/GoogleMap/train/images/target_domain_000237.jpg",  
    # GoogleMap
    "/data/zfx/datasets/CrowdAI/test_images/000000000041.jpg",
    "/data/zfx/datasets/LoveDA/Train/Urban/images_png/2147.png",  # 简单大型建筑物
    "/data/zfx/datasets/LoveDA/Train/Urban/images_png/1474.png",  # 稍复杂小建筑物
    "/data/zfx/datasets/LoveDA/Train/Urban/images_png/1630.png",  # 复杂立体建筑物
]

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector (UDA)', add_help=False)
    
    # === [UDA 新增参数] ===
    parser.add_argument('--source_path', type=str, required=True, help="Path to Source Dataset (e.g., CrowdAI)")
    parser.add_argument('--target_path', type=str, required=True, help="Path to Target Dataset (e.g., Jilin-1)")
    parser.add_argument('--use_mae', action='store_true', help="Enable MAE branch for UDA")
    parser.add_argument('--lambda_unsup', default=0.4, type=float, help="Weight for unsupervised loss")
    parser.add_argument('--lambda_mae', default=1.0, type=float, help="Weight for MAE reconstruction loss")
    parser.add_argument('--mask_ratio', default=0.75, type=float, help="Mask ratio for MAE")
    parser.add_argument('--pseudo_corner_thresh', default=0.45, type=float, help="教师伪标签角点分数阈值")
    parser.add_argument('--pseudo_corner_nms_thresh', default=10.0, type=float, help="教师伪标签角点NMS距离阈值（像素）")
    parser.add_argument('--disable_pseudo_corner_nms', action='store_true', help="构建教师伪标签时禁用角点NMS")
    parser.add_argument('--pseudo_thr_init', default=0.34, type=float, help="自适应伪标签阈值EMA初始值")
    parser.add_argument('--pseudo_thr_min', default=0.35, type=float, help="自适应伪标签阈值最小值")
    parser.add_argument('--pseudo_thr_max', default=0.60, type=float, help="自适应伪标签阈值最大值")
    parser.add_argument('--pseudo_thr_quantile', default=0.94, type=float, help="目标域置信度分位数，用于估计目标阈值")
    parser.add_argument('--pseudo_thr_target_ema_momentum', default=0.95, type=float, help="目标域分位数阈值EMA动量")
    parser.add_argument('--pseudo_thr_source_weight', default=0.20, type=float, help="融合阈值中source动态阈值权重")
    parser.add_argument('--pseudo_thr_target_weight', default=0.80, type=float, help="融合阈值中target分位数EMA权重")
    parser.add_argument('--pseudo_topk', default=25, type=int, help="每张图最多保留的伪标签数量")
    parser.add_argument('--pseudo_min_box_area', default=0.004, type=float, help="伪标签最小框面积（归一化wh面积）")
    parser.add_argument('--disable_pseudo_polygon_nms', action='store_true', help="构建教师伪标签时禁用多边形NMS")
    parser.add_argument('--pseudo_polygon_nms_iou', default=0.30, type=float, help="教师伪标签多边形NMS的IoU阈值")
    parser.add_argument('--pseudo_polygon_nms_downsample', default=4, type=int, help="多边形IoU栅格化降采样倍率")
    parser.add_argument('--disable_pseudo_self_intersection_repair', action='store_true', help="构建教师伪标签时禁用多边形自交修复")
    parser.add_argument('--burn_in_epochs', default=5, type=int,
                        help="Burn-in epochs: train with Source supervision + Target MAE only")
    parser.add_argument('--retrain_interval', default=0, type=int,
                        help="[兼容旧参数] 若>0则覆盖阶段1重训间隔")
    parser.add_argument('--retrain_stage1_epochs', default=60, type=int,
                        help="第二阶段前期长度（按 stage2_epoch 计）")
    parser.add_argument('--retrain_stage1_interval', default=15, type=int,
                        help="第二阶段前期重训间隔")
    parser.add_argument('--retrain_stage2_epochs', default=60, type=int,
                        help="第二阶段中期长度（按 stage2_epoch 计）")
    parser.add_argument('--retrain_stage2_interval', default=20, type=int,
                        help="第二阶段中期重训间隔")
    parser.add_argument('--retrain_stage3_interval', default=30, type=int,
                        help="第二阶段后期重训间隔（0表示关闭）")
    parser.add_argument('--retrain_force_resume', action='store_true',
                        help="重训时强制使用--resume权重，不使用theta_mask_clean")
    parser.add_argument('--disable_retrain_gate', action='store_true',
                        help="禁用重训门控（只按阶段周期触发）")
    parser.add_argument('--retrain_gate_conf_mean_max', default=0.26, type=float,
                        help="门控阈值：上一轮 avg_conf <= 该值时允许重训")
    parser.add_argument('--retrain_gate_conf_kept_max', default=0.55, type=float,
                        help="门控阈值：上一轮 avg_conf_kept <= 该值时允许重训")
    parser.add_argument('--disable_retrain_event_trigger', action='store_true',
                        help="禁用重训事件触发（仅按阶段周期触发）")
    parser.add_argument('--retrain_event_conf_drop', default=0.02, type=float,
                        help="事件触发阈值：avg_conf 单轮下降超过该值时触发")
    parser.add_argument('--retrain_event_kept_drop', default=0.03, type=float,
                        help="事件触发阈值：avg_conf_kept 单轮下降超过该值时触发")
    parser.add_argument('--retrain_event_thr_drop', default=0.025, type=float,
                        help="事件触发阈值：effective_threshold 单轮下降超过该值时触发")
    parser.add_argument('--retrain_cooldown_epochs', default=6, type=int,
                        help="两次重训之间的最小冷却epoch数")
    # ====================

    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    
    # 多边形 Loss 系数
    parser.add_argument('--poly_coord_loss_coef', default=5.0, type=float)
    parser.add_argument('--poly_corner_loss_coef', default=1.0, type=float)

    # dataset parameters (legacy, will be overridden by source/target path)
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser

def visualize_training_progress(model, samples, output_dir, epoch, device, pseudo_thr=0.5):
    """
    在训练过程中可视化固定的样本 (Teacher 模型效果)
    """
    model.eval()
    inputs = samples.tensors.to(device)
    with torch.no_grad():
        outputs = model(inputs)
    
    pred_logits = outputs['pred_logits']
    # 优先使用演化后的结果
    pred_polys = outputs.get('pred_polys_evolve_1', outputs['pred_polys_init'])
    pred_corners = outputs.get('pred_vtx_logits_evolve_1', outputs.get('pred_corners_init', None))
    
    pixel_mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(3, 1, 1)
    
    batch_size = inputs.shape[0]
    fig, axs = plt.subplots(1, batch_size, figsize=(batch_size * 6, 6))
    if batch_size == 1: axs = [axs]
    
    for idx in range(batch_size):
        ax = axs[idx]
        img = inputs[idx] * pixel_std + pixel_mean
        img = img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        h, w = img.shape[:2]
        
        ax.imshow(img)
        
        probs = pred_logits[idx].sigmoid()
        scores = probs.squeeze(-1)
        
        # 使用当前可视化阈值进行筛选
        keep = scores > pseudo_thr
        
        valid_polys = pred_polys[idx][keep]
        valid_corners = pred_corners[idx][keep].sigmoid()
        
        for i in range(len(valid_polys)):
            poly = valid_polys[i].cpu().numpy()
            corner = valid_corners[i].cpu().numpy()
            
            poly[:, 0] *= w
            poly[:, 1] *= h
            
            poly_closed = np.concatenate([poly, poly[0:1]], axis=0)
            ax.plot(poly_closed[:, 0], poly_closed[:, 1], c='cyan', linewidth=1.5)
            
            is_corner = corner > 0.5
            if is_corner.any():
                ax.scatter(poly[is_corner, 0], poly[is_corner, 1], c='red', s=4, zorder=10)

        ax.axis('off')
        ax.set_title(f"Epoch {epoch} - Teacher Pred (thr={pseudo_thr:.3f})")

    thr_tag = f"{pseudo_thr:.3f}".replace('.', 'p')
    save_path = Path(output_dir) / f"vis_epoch_{epoch:03d}_thr_{thr_tag}.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def load_custom_vis_samples(image_paths, dataset):
    if not image_paths:
        return None
    tensors = []
    transforms = getattr(dataset, "_transforms", None)
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            if transforms is not None:
                img, _ = transforms(img, {})
            tensors.append(img)
        except Exception as e:
            print(f"Skipping visualization image {p}: {e}")
            
    if not tensors:
        return None
    return nested_tensor_from_tensor_list(tensors)

def selective_retraining(student_model, source_checkpoint_path):
    """
    [MRT 核心策略] 选择性重训练机制
    
    逻辑：
    1. 加载源域预训练权重 (Clean Source Weights)。
    2. 将 Student 模型的 Backbone 和 Transformer Encoder 重置为源域权重。
    3. 保留 Student 模型的 Decoder 和 Prediction Heads (它们包含了对 Target 域的适应知识)。
    
    这相当于：保留“怎么检测物体”的知识(Decoder)，重置“怎么看图”的知识(Encoder)，
    防止 Encoder 对错误的伪标签过拟合。
    """
    if not source_checkpoint_path or not os.path.exists(source_checkpoint_path):
        print(f"⚠️ [Warning] Cannot find source checkpoint at {source_checkpoint_path}. Skipping selective retraining.")
        return

    print(f"🔄 [Selective Retraining] Loading clean weights from {source_checkpoint_path}...")
    
    # 1. 加载纯净的源域权重
    checkpoint = torch.load(source_checkpoint_path, map_location='cpu', weights_only=False)
    src_state_dict = checkpoint['model']
    
    # 2. 获取当前 Student 的权重
    student_state_dict = student_model.state_dict()
    
    # 3. 筛选并覆盖
    reset_count = 0
    keys_to_reset = []
    
    for key in student_state_dict.keys():
        # MRT 论文结论: 重置 Backbone 和 Encoder 效果最好
        # 关键词匹配: 'backbone', 'transformer.encoder', 'input_proj'
        if ('backbone' in key or 
            'transformer.encoder' in key or 
            'input_proj' in key):
            
            if key in src_state_dict:
                student_state_dict[key] = src_state_dict[key]
                keys_to_reset.append(key)
                reset_count += 1
            else:
                # 这种情况很少见，除非模型结构变了
                pass
    
    # 4. 加载回模型
    student_model.load_state_dict(student_state_dict, strict=False)
    print(f"✅ [Selective Retraining] Successfully reset {reset_count} parameters (Backbone + Encoder). Decoder kept.")


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # Fix seed
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 1. 构建 Student 模型
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    # 2. 构建 Teacher 模型 (完全复制 Student)
    teacher_model = copy.deepcopy(model)
    # Teacher 不反向传播
    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # 3. 加载数据集 (关键: 分别加载 Source 和 Target)
    # 技巧: 临时修改 args.coco_path 来复用 build_dataset
    
    # 3.1 加载 Source (CrowdAI)
    print(f"Loading Source Dataset from: {args.source_path}")
    args.coco_path = args.source_path
    dataset_source_train = build_dataset(image_set='train', args=args)
    
    # 3.2 加载 Target (Jilin-1)
    print(f"Loading Target Dataset from: {args.target_path}")
    args.coco_path = args.target_path
    # 即使 Target 没有标签，我们也可以加载它。CocoDetection 会寻找 json 文件。
    # 确保你的 Target 文件夹里有一个 dummy json (即使是空的 annotations 列表) 
    # 或者如果 build_dataset 支持纯图片文件夹请相应调整。
    # 这里假设 Jilin-1 遵循 COCO 格式。
    dataset_target_train = build_dataset(image_set='train', args=args)
    # Target 评估集（如无独立 val，则用 train 顺序遍历）
    dataset_target_eval = dataset_target_train
    
    # 3.3 加载验证集 (Source 的验证集，用于监控性能)
    # 也可以加载 Target 的验证集如果有的话
    print(f"Loading Validation Dataset from: {args.source_path}")
    args.coco_path = args.source_path
    dataset_val = build_dataset(image_set='val', args=args)

    # 4. 构建 DataLoaders
    if args.distributed:
        if args.cache_mode:
            sampler_source = samplers.NodeDistributedSampler(dataset_source_train)
            sampler_target = samplers.NodeDistributedSampler(dataset_target_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
            sampler_target_eval = samplers.NodeDistributedSampler(dataset_target_eval, shuffle=False)
        else:
            sampler_source = samplers.DistributedSampler(dataset_source_train)
            sampler_target = samplers.DistributedSampler(dataset_target_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
            sampler_target_eval = samplers.DistributedSampler(dataset_target_eval, shuffle=False)
    else:
        sampler_source = torch.utils.data.RandomSampler(dataset_source_train)
        sampler_target = torch.utils.data.RandomSampler(dataset_target_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_target_eval = torch.utils.data.SequentialSampler(dataset_target_eval)

    batch_sampler_source = torch.utils.data.BatchSampler(sampler_source, args.batch_size, drop_last=True)
    batch_sampler_target = torch.utils.data.BatchSampler(sampler_target, args.batch_size, drop_last=True)

    data_loader_source = DataLoader(dataset_source_train, batch_sampler=batch_sampler_source,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=False)
    
    data_loader_target = DataLoader(dataset_target_train, batch_sampler=batch_sampler_target,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=False)

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=False)
    data_loader_target_eval = DataLoader(dataset_target_eval, args.batch_size, sampler=sampler_target_eval,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=False)

    # 可视化样本准备
    vis_samples = None
    if args.output_dir:
        if CUSTOM_VIS_IMAGE_PATHS:
            print("Loading custom images for visualization...")
            vis_samples = load_custom_vis_samples(CUSTOM_VIS_IMAGE_PATHS, dataset_val)
        if vis_samples is None:
            try:
                vis_samples, _ = next(iter(data_loader_val))
            except:
                pass

    # 5. 优化器设置 (只优化 Student)
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
        
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.5)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # 数据集 API
    if args.dataset_file == "coco_panoptic":
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)
    base_ds_target = get_coco_api_from_dataset(dataset_target_eval)

    # 6. 加载预训练权重 (Student & Teacher)
    # 如果指定了 resume，则加载 checkpoint
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            
        state_dict = checkpoint['model']
        
        # 过滤不匹配的键
        keys_to_remove = [k for k in state_dict.keys() if k.startswith("class_embed")]
        for k in keys_to_remove:
            del state_dict[k]
            
        # 加载到 Student
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(state_dict, strict=False)
        print(f"Student Load - Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
        
        # 加载到 Teacher (保持一致)
        teacher_model.load_state_dict(state_dict, strict=False)
        
        # -------------------------------------------------------------------------
        # 修改部分：健壮的权重加载逻辑
        # -------------------------------------------------------------------------
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            try:
                # 尝试加载优化器
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                # 如果成功加载，说明是“断点续训”，恢复 Epoch
                args.start_epoch = checkpoint['epoch'] + 1
                print(f"✅ 成功恢复优化器状态，从 Epoch {args.start_epoch} 继续训练。")
            except ValueError as e:
                # 如果加载失败（通常是因为模型结构变了，比如加了 MAE），则忽略优化器状态
                print(f"⚠️ 警告: 优化器加载失败 (参数不匹配)。")
                print(f"   原因: {e}")
                print(f"   处理: 忽略旧优化器状态，使用新初始化的优化器从 Epoch 0 开始 UDA 训练。")
                # 保持 args.start_epoch = 0
        # -------------------------------------------------------------------------

    output_dir = Path(args.output_dir)
    writer = None
    if args.output_dir and utils.is_main_process():
        writer = SummaryWriter(log_dir=str(output_dir))
        run_meta_path = output_dir / "A.txt"
        launch_cmd = " ".join(shlex.quote(x) for x in sys.argv)
        key_hparams = {
            "source_path": args.source_path,
            "target_path": args.target_path,
            "resume": args.resume,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lr_drop": args.lr_drop,
            "burn_in_epochs": args.burn_in_epochs,
            "lambda_unsup": args.lambda_unsup,
            "lambda_mae": args.lambda_mae,
            "retrain_force_resume": args.retrain_force_resume,
            "retrain_stage1_epochs": args.retrain_stage1_epochs,
            "retrain_stage1_interval": args.retrain_stage1_interval,
            "retrain_stage2_epochs": args.retrain_stage2_epochs,
            "retrain_stage2_interval": args.retrain_stage2_interval,
            "retrain_stage3_interval": args.retrain_stage3_interval,
            "pseudo_thr_init": args.pseudo_thr_init,
            "pseudo_thr_min": args.pseudo_thr_min,
            "pseudo_thr_max": args.pseudo_thr_max,
            "pseudo_thr_quantile": args.pseudo_thr_quantile,
            "pseudo_thr_target_ema_momentum": args.pseudo_thr_target_ema_momentum,
            "pseudo_thr_source_weight": args.pseudo_thr_source_weight,
            "pseudo_thr_target_weight": args.pseudo_thr_target_weight,
            "pseudo_topk": args.pseudo_topk,
            "pseudo_min_box_area": args.pseudo_min_box_area,
        }
        with run_meta_path.open("w", encoding="utf-8") as f:
            f.write(f"run_start_time: {datetime.datetime.now().isoformat()}\n")
            f.write(f"git_sha: {utils.get_sha()}\n")
            f.write(f"cuda_visible_devices: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}\n")
            f.write(f"launch_command: {launch_cmd}\n\n")
            f.write("key_hyperparameters:\n")
            f.write(json.dumps(key_hparams, ensure_ascii=False, indent=2) + "\n\n")
            f.write("all_args:\n")
            f.write(json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n")

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        return

    # =========================================================================
    # Training Loop
    # =========================================================================
    print("Start UDA training")
    start_time = time.time()
    
    if args.retrain_interval > 0:
        args.retrain_stage1_interval = args.retrain_interval
    clean_ckpt_path = output_dir / 'theta_mask_clean.pth' if args.output_dir else None
    best_ckpt_path = output_dir / 'checkpoint_target_best.pth' if args.output_dir else None
    prev_train_stats = None
    prev_prev_train_stats = None
    last_retrain_epoch = -10**9
    max_avg_confidence_kept = 0.0

    def get_retrain_interval(stage2_epoch):
        if stage2_epoch <= 0:
            return 0
        if stage2_epoch <= args.retrain_stage1_epochs:
            return args.retrain_stage1_interval
        if stage2_epoch <= (args.retrain_stage1_epochs + args.retrain_stage2_epochs):
            return args.retrain_stage2_interval
        return args.retrain_stage3_interval
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_source.set_epoch(epoch)
            sampler_target.set_epoch(epoch)

        retrain_due = False
        retrain_schedule_due = False
        retrain_event_due = False
        retrain_cooldown_ok = True
        retrain_triggered = False
        retrain_gate_pass = False
        retrain_interval_curr = 0
        retrain_gate_reason = "not_checked"
        retrain_action = "not_due"

        # ---------------------------------------------------------------------
        # [MRT] Selective Retraining Trigger (阶段化 + 门控)
        # 逻辑：
        # 1. 仅在第二阶段触发（stage2_epoch > 0）
        # 2. 阶段化间隔：前期/中期/后期
        # 3. 门控触发：仅当上一轮置信度质量下降时才执行重训
        # ---------------------------------------------------------------------
        stage2_epoch = epoch - args.burn_in_epochs
        retrain_interval_curr = get_retrain_interval(stage2_epoch)
        schedule_due = retrain_interval_curr > 0 and stage2_epoch % retrain_interval_curr == 0
        retrain_schedule_due = schedule_due

        event_due = False
        event_reason = "no_event"
        if not args.disable_retrain_event_trigger and max_avg_confidence_kept > 0.0 and prev_train_stats is not None:
            prev_kept = float(prev_train_stats.get('pseudo_conf_kept_mean', float('nan')))
            
            # 使用历史最高点计算回撤
            if not np.isnan(prev_kept):
                kept_drop = max_avg_confidence_kept - prev_kept
                if kept_drop >= args.retrain_event_kept_drop:
                    event_due = True
                    event_reason = (
                        f"event_drop(max_kept={max_avg_confidence_kept:.4f}, curr_kept={prev_kept:.4f}, "
                        f"drop={kept_drop:.4f} >= {args.retrain_event_kept_drop:.4f})"
                    )

        retrain_event_due = event_due
        retrain_cooldown_ok = (epoch - last_retrain_epoch) >= args.retrain_cooldown_epochs
        retrain_due = schedule_due or event_due

        gate_pass = True
        gate_reason = "gate disabled"
        if not args.disable_retrain_gate:
            gate_pass = False
            gate_reason = "no previous stats"
            if prev_train_stats is not None:
                prev_conf_mean = float(prev_train_stats.get('pseudo_conf_mean', float('nan')))
                prev_conf_kept = float(prev_train_stats.get('pseudo_conf_kept', float('nan')))
                low_conf = (not np.isnan(prev_conf_mean)) and (prev_conf_mean <= args.retrain_gate_conf_mean_max)
                low_kept = (not np.isnan(prev_conf_kept)) and (prev_conf_kept <= args.retrain_gate_conf_kept_max)
                gate_pass = low_conf or low_kept
                gate_reason = (
                    f"prev_avg_conf={prev_conf_mean:.4f}, prev_avg_conf_kept={prev_conf_kept:.4f}, "
                    f"thresholds=({args.retrain_gate_conf_mean_max:.4f}, {args.retrain_gate_conf_kept_max:.4f})"
                )

        if retrain_due:
            if not retrain_cooldown_ok:
                retrain_gate_pass = False
                retrain_action = "skipped_by_cooldown"
                retrain_gate_reason = f"cooldown_active(last_retrain_epoch={last_retrain_epoch}, cooldown={args.retrain_cooldown_epochs})"
                print(
                    f"[Retrain] SKIP at Epoch {epoch} due to cooldown. "
                    f"last={last_retrain_epoch}, cooldown={args.retrain_cooldown_epochs}"
                )
            elif event_due:
                retrain_triggered = True
                retrain_gate_pass = True
                retrain_action = "triggered"
                print(
                    f"\n⚡ Triggering Selective Retraining at Epoch {epoch} "
                    f"(event-based, stage2_epoch={stage2_epoch}) ⚡"
                )
                print(f"[RetrainEvent] {event_reason}")
                if args.retrain_force_resume:
                    retrain_path = args.resume
                else:
                    if best_ckpt_path is not None and best_ckpt_path.exists():
                        retrain_path = str(best_ckpt_path)
                    else:
                        retrain_path = str(clean_ckpt_path) if (clean_ckpt_path is not None and clean_ckpt_path.exists()) else args.resume
                print(f"[Fallback] Using checkpoint: {retrain_path}")
                selective_retraining(model_without_ddp, retrain_path)
                last_retrain_epoch = epoch
            elif gate_pass:
                retrain_triggered = True
                retrain_gate_pass = True
                retrain_action = "triggered"
                print(
                    f"\n⚡ Triggering Selective Retraining at Epoch {epoch} "
                    f"(stage2_epoch={stage2_epoch}, interval={retrain_interval_curr}) ⚡"
                )
                print(f"[RetrainGate] PASS: {gate_reason}")
                # 注意：传入 model_without_ddp 以避免 DDP 包装器的前缀问题
                # 优先加载 Best Model，如果没有才用 Burn-in 阶段产出的 clean 权重
                if args.retrain_force_resume:
                    retrain_path = args.resume
                else:
                    if best_ckpt_path is not None and best_ckpt_path.exists():
                        retrain_path = str(best_ckpt_path)
                    else:
                        retrain_path = str(clean_ckpt_path) if (clean_ckpt_path is not None and clean_ckpt_path.exists()) else args.resume
                print(f"[Fallback] Using checkpoint: {retrain_path}")
                selective_retraining(model_without_ddp, retrain_path)
                last_retrain_epoch = epoch
            else:
                retrain_gate_pass = False
                retrain_action = "skipped_by_gate"
                print(
                    f"[RetrainGate] SKIP at Epoch {epoch} "
                    f"(stage2_epoch={stage2_epoch}, interval={retrain_interval_curr}). {gate_reason}"
                )
        else:
            retrain_gate_pass = gate_pass
            retrain_action = "not_due"
            retrain_gate_reason = gate_reason
        if retrain_action not in {"triggered_by_event", "skipped_by_cooldown"}:
            retrain_gate_reason = gate_reason
        # ---------------------------------------------------------------------

        # 调用训练函数 (使用我们刚刚优化过的 engine.py)
        train_stats = train_one_epoch(
            student=model,
            teacher=teacher_model,
            criterion=criterion,
            data_loader_source=data_loader_source,
            data_loader_target=data_loader_target,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_norm=args.clip_max_norm,
            args=args 
        )
        prev_prev_train_stats = prev_train_stats
        prev_train_stats = train_stats
        epoch_pseudo_conf_mean = float(train_stats.get('pseudo_conf_mean', float('nan')))
        epoch_pseudo_conf_kept_mean = float(train_stats.get('pseudo_conf_kept_mean', float('nan')))
        raw_pseudo_thr = float(train_stats.get('pseudo_thr_raw_mean', float('nan')))
        effective_pseudo_thr = float(train_stats.get('pseudo_thr_effective_mean', float('nan')))
        target_ema_pseudo_thr = float(train_stats.get('pseudo_thr_target_ema_mean', float('nan')))
        quantile_pseudo_thr = float(train_stats.get('pseudo_thr_quantile_mean', float('nan')))

        # =====================================================================
        # [新增] 动态保存 Target 域历史最佳权重 (依据 pseudo_conf_kept_mean)
        # =====================================================================
        if epoch_pseudo_conf_kept_mean and not np.isnan(epoch_pseudo_conf_kept_mean):
            if epoch_pseudo_conf_kept_mean > max_avg_confidence_kept:
                max_avg_confidence_kept = epoch_pseudo_conf_kept_mean
                if args.output_dir and utils.is_main_process() and best_ckpt_path is not None:
                    print(f"🌟 [New Best Target Model] conf_kept reached {max_avg_confidence_kept:.6f}. Saving to {best_ckpt_path}")
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, best_ckpt_path)

        print(
            f"[PseudoThr][Epoch {epoch}] raw={raw_pseudo_thr:.6f}, "
            f"target_ema={target_ema_pseudo_thr:.6f}, q_thr={quantile_pseudo_thr:.6f}, "
            f"effective={effective_pseudo_thr:.6f}, avg_conf={epoch_pseudo_conf_mean:.6f}, "
            f"avg_conf_kept={epoch_pseudo_conf_kept_mean:.6f}"
        )

        # Burn-in 结束：保存 clean 权重，并将 Teacher 与 clean Student 对齐
        if args.output_dir and epoch == (args.burn_in_epochs - 1):
            print(f"\n💾 Saving clean burn-in weights to {clean_ckpt_path}")
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'epoch': epoch,
                'args': args,
                'tag': 'theta_mask_clean'
            }, clean_ckpt_path)
            teacher_model.load_state_dict(model_without_ddp.state_dict(), strict=False)
            print("✅ Teacher synced from theta_mask_clean.")
        
        lr_scheduler.step()

        # 可视化 (使用 Teacher 模型查看伪标签效果)
        if args.output_dir and vis_samples is not None and utils.is_main_process():
            try:
                vis_pseudo_thr = effective_pseudo_thr if not np.isnan(effective_pseudo_thr) else 0.5
                visualize_training_progress(
                    teacher_model,
                    vis_samples,
                    args.output_dir,
                    epoch,
                    device,
                    pseudo_thr=vis_pseudo_thr,
                )
            except Exception as e:
                print(f"Vis failed: {e}")

        # Checkpointing
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        # Eval (使用 Teacher 模型评估，通常 Teacher 泛化更好)
        test_stats = {}
        test_bbox_ap = None
        test_bbox_ap50 = None
        test_bbox_ap75 = None
        # 评估周期 评估间隔周期
        if (epoch + 1) % 120 == 0 or (epoch + 1) == args.epochs:
            print(f"Evaluating Teacher Model at epoch {epoch}...")
            test_stats, coco_evaluator = evaluate(
                teacher_model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
            )
            if 'coco_eval_bbox' in test_stats:
                test_bbox_ap = test_stats['coco_eval_bbox'][0]
                if len(test_stats['coco_eval_bbox']) > 2:
                    test_bbox_ap50 = test_stats['coco_eval_bbox'][1]
                    test_bbox_ap75 = test_stats['coco_eval_bbox'][2]

        # 每轮记录 Teacher 在目标域上的表现（使用 target_eval loader）
        target_test_stats = {}
        target_bbox_ap = None
        target_bbox_ap50 = None
        target_bbox_ap75 = None
        target_has_ann = False
        try:
            target_has_ann = hasattr(base_ds_target, "dataset") and \
                             len(base_ds_target.dataset.get("annotations", [])) > 0
        except Exception:
            target_has_ann = False

        if target_has_ann:
            try:
                target_test_stats, _ = evaluate(
                    teacher_model, criterion, postprocessors, data_loader_target_eval, base_ds_target, device, args.output_dir
                )
                if 'coco_eval_bbox' in target_test_stats:
                    target_bbox_ap = target_test_stats['coco_eval_bbox'][0]
                    if len(target_test_stats['coco_eval_bbox']) > 2:
                        target_bbox_ap50 = target_test_stats['coco_eval_bbox'][1]
                        target_bbox_ap75 = target_test_stats['coco_eval_bbox'][2]
            except Exception as e:
                print(f"Target eval failed: {e}")
        else:
            print("Target eval skipped: no annotations found in target dataset.")

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     **{f'target_{k}': v for k, v in target_test_stats.items()},
                     'test_bbox_mAP': test_bbox_ap,
                     'test_bbox_AP50': test_bbox_ap50,
                     'test_bbox_AP75': test_bbox_ap75,
                     'target_bbox_mAP': target_bbox_ap,
                     'target_bbox_AP50': target_bbox_ap50,
                     'target_bbox_AP75': target_bbox_ap75,
                     'retrain_due': retrain_due,
                     'retrain_schedule_due': retrain_schedule_due,
                     'retrain_event_due': retrain_event_due,
                     'retrain_cooldown_ok': retrain_cooldown_ok,
                     'retrain_triggered': retrain_triggered,
                     'retrain_stage2_epoch': stage2_epoch,
                     'retrain_interval_curr': retrain_interval_curr,
                     'retrain_gate_pass': retrain_gate_pass,
                     'retrain_action': retrain_action,
                     'retrain_gate_reason': retrain_gate_reason,
                     'pseudo_thr_raw': raw_pseudo_thr,
                     'pseudo_thr_target_ema': target_ema_pseudo_thr,
                     'pseudo_thr_quantile': quantile_pseudo_thr,
                     'pseudo_thr_effective': effective_pseudo_thr,
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            with (output_dir / "pseudo_threshold_log.txt").open("a") as f:
                if f.tell() == 0:
                    f.write(
                        "epoch(轮次)\traw_threshold(源域动态阈值)\ttarget_ema_threshold(目标域EMA阈值)"
                        "\tquantile_threshold(目标域分位数阈值)\teffective_threshold(生效阈值)"
                        "\tavg_confidence(平均置信度)\tavg_confidence_kept(保留伪标签平均置信度)"
                        "\tretrain_due(是否触发重训检查)\tretrain_schedule_due(是否周期触发)"
                        "\tretrain_event_due(是否事件触发)\tretrain_cooldown_ok(是否通过冷却)"
                        "\tretrain_triggered(是否实际重训)"
                        "\tretrain_stage2_epoch(第二阶段轮次)\tretrain_interval(当前阶段重训间隔)"
                        "\tretrain_gate_pass(门控是否通过)\tretrain_action(重训动作)"
                        "\tretrain_gate_reason(门控原因)\n"
                    )
                raw_str = f"{raw_pseudo_thr:.6f}"
                tgt_ema_str = f"{target_ema_pseudo_thr:.6f}"
                q_str = f"{quantile_pseudo_thr:.6f}"
                eff_str = f"{effective_pseudo_thr:.6f}"
                conf_str = f"{epoch_pseudo_conf_mean:.6f}"
                conf_kept_str = f"{epoch_pseudo_conf_kept_mean:.6f}"
                gate_reason_sanitized = str(retrain_gate_reason).replace('\t', ' ').replace('\n', ' ')
                f.write(
                    f"{epoch}\t{raw_str}\t{tgt_ema_str}\t{q_str}\t{eff_str}\t{conf_str}\t{conf_kept_str}"
                    f"\t{int(retrain_due)}\t{int(retrain_schedule_due)}\t{int(retrain_event_due)}\t{int(retrain_cooldown_ok)}"
                    f"\t{int(retrain_triggered)}\t{stage2_epoch}\t{retrain_interval_curr}"
                    f"\t{int(retrain_gate_pass)}\t{retrain_action}\t{gate_reason_sanitized}\n"
                )
            
            if writer is not None:
                for k, v in log_stats.items():
                    if isinstance(v, (int, float)):
                        writer.add_scalar(k, v, epoch)
                writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR UDA Training', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

    """
        python main.py \
            --source_path "/data/zfx/datasets/CrowdAI" \
            --target_path "/data/zfx/datasets/Jilin-1" \
            --output_dir "/data/zfx/myuda/uda_jilin_run_v2" \
            --resume "weight/checkpoint_crowd.pth" \
            --use_mae \
            --mask_ratio 0.75 \
            --lambda_unsup 1.0 \
            --lambda_mae 1.0 \
            --epochs 50 \
            --lr 2e-4 \
            --lr_drop 40 \
            --batch_size 2 \
            --num_workers 4 \
            --num_queries 300 \
            --poly_coord_loss_coef 5.0 \
            --poly_corner_loss_coef 1.0 \
            --focal_alpha 0.25

        双卡：torchrun --nproc_per_node=2
    """