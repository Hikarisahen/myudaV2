# ------------------------------------------------------------------------
# Deformable DETR (UDA Version: FDA + Mean Teacher + MAE)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import sys
import os
import copy  # [新增] 用于复制 Teacher 模型

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
    # 这里建议替换为吉林一号（目标域）的图片路径，用于观察适应效果
    "/data/zfx/datasets/CrowdAI/test_images/000000000041.jpg",
    "/data/zfx/datasets/Jilin-1/train/images/tile_0_crop_49_64.jpg",
    "/data/zfx/datasets/Jilin-1/train/images/tile_0_crop_68_82.jpg",
    "/data/zfx/datasets/Jilin-1/train/images/tile_0_crop_79_5.jpg",
]

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector (UDA)', add_help=False)
    
    # === [UDA 新增参数] ===
    parser.add_argument('--source_path', type=str, required=True, help="Path to Source Dataset (e.g., CrowdAI)")
    parser.add_argument('--target_path', type=str, required=True, help="Path to Target Dataset (e.g., Jilin-1)")
    parser.add_argument('--use_mae', action='store_true', help="Enable MAE branch for UDA")
    parser.add_argument('--lambda_unsup', default=1.0, type=float, help="Weight for unsupervised loss")
    parser.add_argument('--lambda_mae', default=1.0, type=float, help="Weight for MAE reconstruction loss")
    parser.add_argument('--mask_ratio', default=0.75, type=float, help="Mask ratio for MAE")
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

def visualize_training_progress(model, samples, output_dir, epoch, device):
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
        
        # 可视化阈值设高一点
        keep = scores > 0.5 
        
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
        ax.set_title(f"Epoch {epoch} - Teacher Pred")

    save_path = Path(output_dir) / f"vis_epoch_{epoch:03d}.png"
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
        else:
            sampler_source = samplers.DistributedSampler(dataset_source_train)
            sampler_target = samplers.DistributedSampler(dataset_target_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_source = torch.utils.data.RandomSampler(dataset_source_train)
        sampler_target = torch.utils.data.RandomSampler(dataset_target_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

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

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        return

    # =========================================================================
    # Training Loop
    # =========================================================================
    print("Start UDA training")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_source.set_epoch(epoch)
            sampler_target.set_epoch(epoch)

        # ---------------------------------------------------------------------
        # Selective Retraining 逻辑 (MRT Paper)
        # 每隔 10 个 Epoch，将 Student 的 Encoder 重置为当前 Encoder 的状态
        # (在我们的实现中，Student 一直在做 MAE，所以 Encoder 已经是 Refined 过的)
        # 这里模拟 "Retraining"：可以理解为一次 Checkpoint，或者更激进的重置 Head
        # 为了稳定性，我们这里暂不重置 Head，而是打印日志确认 MAE 正在工作
        # ---------------------------------------------------------------------
        if epoch % 10 == 0 and epoch > 0:
            print(f"[Selective Retraining] Epoch {epoch}: Keeping Encoder weights derived from MAE task.")
            # 如果想要激进的重置，可以解除下面代码的注释：
            # print("Re-initializing Detection Head...")
            # model_without_ddp.class_embed.reset_parameters()
            # model_without_ddp.bbox_embed.reset_parameters()

        # 调用我们修改过的 UDA 训练函数
        train_stats = train_one_epoch(
            student=model,
            teacher=teacher_model, # 传入 Teacher
            criterion=criterion,
            data_loader_source=data_loader_source,
            data_loader_target=data_loader_target,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_norm=args.clip_max_norm,
            args=args # 传入 args 以获取 lambda
        )
        
        lr_scheduler.step()

        # 可视化 (使用 Teacher 模型查看伪标签效果)
        if args.output_dir and vis_samples is not None and utils.is_main_process():
            try:
                visualize_training_progress(teacher_model, vis_samples, args.output_dir, epoch, device)
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
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            print(f"Evaluating Teacher Model at epoch {epoch}...")
            test_stats, coco_evaluator = evaluate(
                teacher_model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
            )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            
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