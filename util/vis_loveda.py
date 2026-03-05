import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

# LoveDA 官方类别 (按论文/常用顺序)
# 0: Background, 1: Building, 2: Road, 3: Water, 4: Barren, 5: Forest, 6: Agriculture
PALETTE: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),          # background - black
    1: (255, 0, 0),        # building - red
    2: (255, 255, 0),      # road - yellow
    3: (0, 0, 255),        # water - blue
    4: (210, 180, 140),    # barren - tan
    5: (34, 139, 34),      # forest - green
    6: (255, 165, 0),      # agriculture - orange
}

def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Map single-channel mask (H,W) of class ids to RGB (H,W,3)."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, rgb in PALETTE.items():
        color[mask == cls_id] = rgb
    return color

def overlay_image(img: Image.Image, mask_rgb: np.ndarray, alpha: float) -> Image.Image:
    """Alpha-blend mask_rgb onto img."""
    img_np = np.array(img.convert("RGB"), dtype=np.float32)
    mask_np = mask_rgb.astype(np.float32)
    blended = (1 - alpha) * img_np + alpha * mask_np
    blended = blended.clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)

def main():
    parser = argparse.ArgumentParser("LoveDA mask visualizer")
    parser.add_argument("--image", help="Path to single PNG image")
    parser.add_argument("--mask", help="Path to single PNG mask (class ids)")
    parser.add_argument("--image_dir", help="Folder of images (filenames must match masks)")
    parser.add_argument("--mask_dir", help="Folder of masks (filenames must match images)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Mask overlay transparency [0,1]")
    parser.add_argument("--save", default=None, help="Output path for single; if batch, use --out_dir")
    parser.add_argument("--out_dir", default=None, help="Output directory for batch; default: image_dir with _overlay suffix")
    args = parser.parse_args()

    # decide single or batch
    is_batch = args.image_dir is not None or args.mask_dir is not None
    if is_batch:
        assert args.image_dir and args.mask_dir, "Batch mode requires both --image_dir and --mask_dir"
        img_dir = Path(args.image_dir)
        mask_dir = Path(args.mask_dir)
        assert img_dir.exists(), f"Image dir not found: {img_dir}"
        assert mask_dir.exists(), f"Mask dir not found: {mask_dir}"

        out_dir = Path(args.out_dir) if args.out_dir else img_dir.with_name(img_dir.name + "_overlay")
        out_dir.mkdir(parents=True, exist_ok=True)

        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        img_paths = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in exts]
        if not img_paths:
            raise ValueError("No images found in image_dir")

        for img_path in img_paths:
            mask_path = mask_dir / img_path.name
            if not mask_path.exists():
                print(f"[Skip] Mask not found for {img_path.name}")
                continue

            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path)
            mask_np = np.array(mask)
            if mask_np.ndim == 3:
                mask_np = mask_np[..., 0]

            mask_rgb = mask_to_color(mask_np)
            overlay = overlay_image(img, mask_rgb, alpha=args.alpha)

            out_path = out_dir / f"{img_path.stem}_overlay.png"
            overlay.save(out_path)
            print(f"Saved: {out_path}")

    else:
        assert args.image and args.mask, "Single mode requires --image and --mask"
        img_path = Path(args.image)
        mask_path = Path(args.mask)
        assert img_path.exists(), f"Image not found: {img_path}"
        assert mask_path.exists(), f"Mask not found: {mask_path}"

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask_np = np.array(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]

        mask_rgb = mask_to_color(mask_np)
        overlay = overlay_image(img, mask_rgb, alpha=args.alpha)

        out_path = Path(args.save) if args.save else img_path.with_name(img_path.stem + "_overlay.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(out_path)
        print(f"Saved overlay: {out_path}")

if __name__ == "__main__":
    main()
