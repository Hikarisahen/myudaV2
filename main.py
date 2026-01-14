# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import sys
import os

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
from engine import evaluate, train_one_epoch
from models import build_model
import matplotlib
matplotlib.use("Agg")  # 强制无GUI后端，避免X11
import matplotlib.pyplot as plt
import numpy as np

# 固定可视化使用的图片路径，按需替换为你想要的文件
CUSTOM_VIS_IMAGE_PATHS = [
    "/home/zfx/datasets/CrowdAI/train/images/000000000042.jpg",  # 训练集密集地
    "/home/zfx/datasets/CrowdAI/test_images/000000000041.jpg",  # 验证集密集地复杂建筑物
    "/home/zfx/datasets/CrowdAI/test_images/000000059046.jpg",  # 验证集密集地
    "/home/zfx/datasets/WHU/test/300594.TIF"  # WHU测试集大建筑物

]


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
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
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


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
    # [新增] 多边形 Loss 系数
    parser.add_argument('--poly_coord_loss_coef', default=5.0, type=float)
    parser.add_argument('--poly_corner_loss_coef', default=1.0, type=float)

    # dataset parameters
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
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser

def visualize_training_progress(model, samples, output_dir, epoch, device):
    """
    在训练过程中可视化固定的样本
    """
    model.eval() # 切换到评估模式
    
    # 准备数据
    inputs = samples.tensors.to(device)
    # 运行推理
    with torch.no_grad():
        outputs = model(inputs)
    
    # 获取预测 (batch_size, num_queries, ...)
    pred_logits = outputs['pred_logits']
    pred_polys = outputs.get('pred_polys_evolve_1', outputs['pred_polys_init'])
    pred_corners = outputs.get('pred_vtx_logits_evolve_1', outputs.get('pred_corners_init', None))
    
    # 反归一化图片 (ImageNet Mean/Std) 用于显示
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    pixel_mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(3, 1, 1)
    
    batch_size = inputs.shape[0]
    
    # 创建画布
    fig, axs = plt.subplots(1, batch_size, figsize=(batch_size * 6, 6))
    if batch_size == 1: axs = [axs]
    
    for idx in range(batch_size):
        ax = axs[idx]
        
        # 1. 还原图片
        img = inputs[idx] * pixel_std + pixel_mean
        img = img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        h, w = img.shape[:2]
        
        ax.imshow(img)
        
        # 2. 筛选预测结果 (Score > 0.5)
        probs = pred_logits[idx].sigmoid()  # [Q, 1]
        scores = probs.squeeze(-1)          # [Q]
        
        keep = scores > 0.5 # 阈值可调
        
        valid_polys = pred_polys[idx][keep]       # [N, 64, 2]
        valid_corners = pred_corners[idx][keep].sigmoid() # [N, 64]
        
        # 3. 画多边形
        for i in range(len(valid_polys)):
            poly = valid_polys[i].cpu().numpy()
            corner = valid_corners[i].cpu().numpy()
            
            # 还原绝对坐标
            poly[:, 0] *= w
            poly[:, 1] *= h
            
            # 画线
            poly_closed = np.concatenate([poly, poly[0:1]], axis=0)
            ax.plot(poly_closed[:, 0], poly_closed[:, 1], c='cyan', linewidth=1.5)
            
            # 画角点 (Score > 0.5)
            is_corner = corner > 0.5
            if is_corner.any():
                ax.scatter(poly[is_corner, 0], poly[is_corner, 1], c='red', s=2, zorder=10)

        ax.axis('off')
        ax.set_title(f"Epoch {epoch} - Sample {idx}")

    # 保存图片
    save_path = Path(output_dir) / f"vis_epoch_{epoch:03d}.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    model.train() # 切回训练模式！非常重要！


def load_custom_vis_samples(image_paths, dataset):
    """将指定路径的图片转成 NestedTensor，沿用验证集的 transforms。"""
    if not image_paths:
        return None
    tensors = []
    transforms = getattr(dataset, "_transforms", None)
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        if transforms is not None:
            img, _ = transforms(img, {})
        tensors.append(img)
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

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=False)  # DDP 多进程环境下，它经常和 Python 的多进程机制冲突导致文件句柄泄露,改为 False
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=False)  # DDP 多进程环境下，它经常和 Python 的多进程机制冲突导致文件句柄泄露,改为 False
    # === [新增] 固定数据用于可视化 ===
    vis_samples = None
    if args.output_dir:
        if CUSTOM_VIS_IMAGE_PATHS:
            print("Loading custom images for visualization...")
            try:
                vis_samples = load_custom_vis_samples(CUSTOM_VIS_IMAGE_PATHS, dataset_val)
                if vis_samples is not None:
                    print(f"Loaded {vis_samples.tensors.shape[0]} custom images for visualization.")
            except Exception as exc:
                print(f"Custom visualization images failed to load: {exc}")
        if vis_samples is None:
            print("Sampling validation images for visualization...")
            try:
                # 取出第一个 batch (通常 batch_size=2 或 4)
                # next(iter()) 返回 (samples, targets)，我们只需要 samples
                vis_samples, _ = next(iter(data_loader_val))
                print(f"Successfully sampled {vis_samples.tensors.shape[0]} images for visualization.")
            except StopIteration:
                print("Warning: Validation dataset is empty, visualization skipped.")
    # ====================================================

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model_without_ddp.named_parameters():
        print(n)

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
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    # gamma=0.1 表示每次降低 10 倍 (默认)
    # gamma=0.5 表示每次降低 2 倍 (例如 2e-4 -> 1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.5)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)

    # === [新增] 初始化 TensorBoard Writer ===
    writer = None
    if args.output_dir and utils.is_main_process():
        # log_dir 指向输出目录
        writer = SummaryWriter(log_dir=str(output_dir))
    # ========================================

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        # 加载checkpoint模型参数
        state_dict = checkpoint['model']

        # ---- 过滤掉 COCO 的分类头（91类）权重，适配 WHU 单类 ----
        keys_to_remove = []
        for k, v in state_dict.items():
            if k.startswith("class_embed"):
                keys_to_remove.append(k)

        # 如果你用了 with_box_refine，class_embed 是 ModuleList，会出现 class_embed.0/1/...
        # 上面 startswith("class_embed") 能全部覆盖
        for k in keys_to_remove:
            del state_dict[k]

        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(state_dict, strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        print("Missing Keys:", missing_keys)
        print("Unexpected Keys:", unexpected_keys)
        # ------------------------------------------------------
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        # 载入优化器和学习率调度器状态（从头训练多边形分支，不需要继承 COCO 的优化器状态；！！如果是续接自己的断点，需要解开注释！！）######
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        #########################################################################################################################
        
        # check the resumed model
        # if not args.eval:
        #     test_stats, coco_evaluator = evaluate(
        #         model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        #     )
    
    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()

        # === [新增] 每个 Epoch 结束调用可视化 ===
        # utils.is_main_process() 确保只在主进程画图，防止多卡训练时重复保存或冲突
        if args.output_dir and vis_samples is not None and utils.is_main_process():
            print(f"Visualizing epoch {epoch}...")
            try:
                visualize_training_progress(
                    model, vis_samples, args.output_dir, epoch, device
                )
            except Exception as e:
                print(f"Visualization failed: {e}")
        # ======================================

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
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

        test_stats = {}
        coco_evaluator = None
        if (epoch + 1) == args.epochs:
            print(f"Running evaluation at the last epoch {epoch}...")
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
            )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            
            # === [新增] 写入 TensorBoard ===
            if writer is not None:
                for k, v in log_stats.items():
                    # 排除非数字类型的字段 (如 epoch 可能已经是数字了，n_params也是)
                    if isinstance(v, (int, float)):
                        writer.add_scalar(k, v, epoch)
                writer.flush()
            # ==============================

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    '''
    使用下面的指令运行脚本(单卡)：
    python -u main.py \
            --coco_path /home/zfx/datasets/WHU \
            --output_dir /home/data/zfx/DETR/init_train \
            --resume weights/r50_deformable_detr-checkpoint.pth \
            --batch_size 4 \
            --epochs 100 \
            --lr 2e-4 \
            --lr_drop 40 \
            --num_queries 300 \
            --dataset_file coco \
            --poly_coord_loss_coef 5.0 \
            --poly_corner_loss_coef 10.0 \
            --num_workers 2

    双卡训练指令：
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 main.py \
            --coco_path /home/zfx/datasets/WHU \
            --output_dir /home/data/zfx/DETR/double_gpu_run \
            --resume weights/r50_deformable_detr-checkpoint.pth \
            --batch_size 2 \
            --epochs 100 \
            --lr 2e-4 \
            --lr_drop 40 \
            --num_queries 300 \
            --dataset_file coco \
            --poly_coord_loss_coef 5.0 \
            --poly_corner_loss_coef 10.0 \
            --num_workers 4

    OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 main.py \
            --coco_path /home/zfx/datasets/WHU \
            --output_dir /home/data/zfx/DETR/whu_poly_dml_ddp2 \
            --resume weights/r50_deformable_detr-checkpoint.pth \
            --dataset_file coco \
            --epochs 100 \
            --lr 2e-4 \
            --lr_backbone 2e-5 \
            --lr_drop 40 \
            --batch_size 2 \
            --num_queries 300 \
            --num_feature_levels 4 \
            --with_box_refine \
            --poly_coord_loss_coef 5.0 \
            --poly_corner_loss_coef 1.0 \
            --num_workers 4
    
    '''
