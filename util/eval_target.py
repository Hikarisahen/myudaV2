"""
在目标域（已人工标注的子集）上评估模型 bbox AP。

直接接收 image_dir 与 ann_file 两个路径，不依赖训练时的 CocoDetection
目录约定，因此适合评估"从 2000 张里挑出来 60 张人工修标"这种小评估集。

仅做检测（bbox）评估，不计算 loss、不解析 cor_cls_poly 等训练专用字段，
因此你的人工标注只需要标准 COCO 字段（bbox / segmentation / area / iscrowd
/ category_id），不需要 64 顶点角点字段。

用法示例：
    python -m util.eval_target \
        --checkpoint weight/checkpoint_wuhan.pth \
        --image_dir /home/zfx/datasets/wuhan/val/images \
        --ann_file  /home/zfx/datasets/wuhan/val/instances_val.json \
        --with_box_refine --num_queries 300 --num_feature_levels 4 \
        --dataset_file coco --source_path "" --target_path ""
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import util.misc as utils
from main import get_args_parser
from datasets.coco import CocoDetection, make_coco_transforms
from datasets.coco_eval import CocoEvaluator
from infer_visualize import load_model


def build_target_dataset(image_dir, ann_file, return_masks):
    dataset = CocoDetection(
        img_folder=image_dir,
        ann_file=ann_file,
        transforms=make_coco_transforms("val"),
        return_masks=return_masks,
        cache_mode=False,
        local_rank=0,
        local_size=1,
    )
    return dataset


@torch.no_grad()
def run_eval(model, postprocessors, data_loader, base_ds, device, iou_types):
    model.eval()
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Target Eval:"

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes)
        if "segm" in postprocessors:
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors["segm"](
                results, outputs, orig_target_sizes, target_sizes
            )
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results)
        }
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    stats = {}
    if "bbox" in iou_types:
        stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
    if "segm" in iou_types:
        stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    return stats


def main():
    parser = get_args_parser()
    for action in parser._actions:
        if action.dest in {"source_path", "target_path"}:
            action.required = False
            action.default = ""

    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pth")
    parser.add_argument("--image_dir", required=True, help="目标域图片目录")
    parser.add_argument("--ann_file", required=True, help="目标域 COCO 格式标注 JSON")
    parser.add_argument("--batch_size_eval", type=int, default=2)
    parser.add_argument("--num_workers_eval", type=int, default=2)
    parser.add_argument("--save_stats", type=str, default=None,
                        help="可选：将 AP 指标存为 JSON 的路径")
    parser.add_argument("--use_teacher", action="store_true",
                        help="加载 checkpoint['teacher_model']（默认）；与训练时 best 选择口径一致。")
    parser.add_argument("--use_student", action="store_true",
                        help="强制加载 checkpoint['model']（student）。仅用于对比 student/teacher 差异。")

    args = parser.parse_args()
    args.device = torch.device(args.device)

    # 构建模型（含 postprocessors）。load_model 已处理 class_embed 形状不匹配。
    from models import build_model
    model, _criterion, postprocessors = build_model(args)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # 默认加载 teacher（与训练时 best 选择口径一致）。
    # 若 ckpt 没有 teacher 或显式指定 --use_student，则回退到 student。
    if args.use_student:
        which = 'student (forced by --use_student)'
        state_dict = checkpoint["model"]
    elif 'teacher_model' in checkpoint:
        which = 'teacher_model'
        state_dict = checkpoint["teacher_model"]
    else:
        which = 'student (no teacher_model in ckpt)'
        state_dict = checkpoint["model"]
    print(f"[eval_target] Loading weights from: {which}")

    model_state_dict = model.state_dict()
    drop = [k for k, v in state_dict.items()
            if k in model_state_dict and v.shape != model_state_dict[k].shape]
    for k in drop:
        print(f"Drop shape-mismatched key from checkpoint: {k}")
        del state_dict[k]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
    model.to(args.device)
    model.eval()

    return_masks = "segm" in postprocessors
    dataset = build_target_dataset(args.image_dir, args.ann_file, return_masks)
    print(f"Loaded {len(dataset)} images from {args.image_dir}")
    print(f"Annotations: {args.ann_file}")

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size_eval,
        shuffle=False,
        num_workers=args.num_workers_eval,
        collate_fn=utils.collate_fn,
        drop_last=False,
    )

    base_ds = dataset.coco
    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors)
    if not iou_types:
        iou_types = ("bbox",)

    stats = run_eval(model, postprocessors, data_loader, base_ds, args.device, iou_types)

    print("\n========== Target Domain Evaluation ==========")
    if "coco_eval_bbox" in stats:
        b = stats["coco_eval_bbox"]
        print("BBox metrics (COCO):")
        print(f"  AP        @[ IoU=0.50:0.95 | area=all   | maxDets=100 ] = {b[0]:.4f}")
        print(f"  AP        @[ IoU=0.50      | area=all   | maxDets=100 ] = {b[1]:.4f}")
        print(f"  AP        @[ IoU=0.75      | area=all   | maxDets=100 ] = {b[2]:.4f}")
        print(f"  AP        @[ IoU=0.50:0.95 | area=small | maxDets=100 ] = {b[3]:.4f}")
        print(f"  AP        @[ IoU=0.50:0.95 | area=medium| maxDets=100 ] = {b[4]:.4f}")
        print(f"  AP        @[ IoU=0.50:0.95 | area=large | maxDets=100 ] = {b[5]:.4f}")
        print(f"  AR        @[ IoU=0.50:0.95 | maxDets=100              ] = {b[8]:.4f}")
    if "coco_eval_masks" in stats:
        m = stats["coco_eval_masks"]
        print("Segm metrics (COCO):")
        print(f"  AP        @[ IoU=0.50:0.95 ] = {m[0]:.4f}")
        print(f"  AP        @[ IoU=0.50      ] = {m[1]:.4f}")
        print(f"  AP        @[ IoU=0.75      ] = {m[2]:.4f}")

    if args.save_stats:
        out = Path(args.save_stats)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"\nStats saved to {out}")


if __name__ == "__main__":
    main()
