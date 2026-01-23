import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch


def FDA_source_to_target(src_img: torch.Tensor, tgt_img: torch.Tensor, beta: float = 0.01) -> torch.Tensor:
    """
    Minimal FDA: replace low-frequency amplitude of src with that of tgt.
    src_img, tgt_img: [B, 3, H, W], float in [0,1]
    beta: ratio of low-frequency area (e.g., 0.09 for 9%).
    """
    with torch.no_grad():
        fft_src = torch.fft.rfft2(src_img.clone(), dim=(-2, -1))
        fft_tgt = torch.fft.rfft2(tgt_img.clone(), dim=(-2, -1))

        amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)
        amp_tgt = torch.abs(fft_tgt)

        _, _, h, w_half = amp_src.shape  # rfft: last dim is W/2+1
        b_h = int(h * beta)
        b_w = int(w_half * beta)

        amp_src[..., :b_h, :b_w] = amp_tgt[..., :b_h, :b_w]
        amp_src[..., -b_h:, :b_w] = amp_tgt[..., -b_h:, :b_w]

        fft_src_mutated = torch.polar(amp_src, pha_src)
        out = torch.fft.irfft2(fft_src_mutated, s=(h, (w_half - 1) * 2), dim=(-2, -1))
        return out.clamp(0, 1)


def load_image(path: str, size_hw=None) -> torch.Tensor:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Fail to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size_hw is not None:
        img = cv2.resize(img, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img


def tensor_to_bgr_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    x = (x * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description="Visualize FDA style transfer")
    parser.add_argument('--src', required=True, help='Source image path')
    parser.add_argument('--tgt', required=True, help='Target image path')
    parser.add_argument('--beta', type=float, default=0.09, help='Low-frequency ratio')
    parser.add_argument('--output_dir', default='./vis_out_fda', help='Where to save outputs')
    args = parser.parse_args()

    src = load_image(args.src)
    tgt = load_image(args.tgt, size_hw=src.shape[-2:])

    fda = FDA_source_to_target(src, tgt, beta=args.beta)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / 'src.png'), tensor_to_bgr_uint8(src))
    cv2.imwrite(str(out_dir / 'tgt_resized.png'), tensor_to_bgr_uint8(tgt))
    cv2.imwrite(str(out_dir / 'fda.png'), tensor_to_bgr_uint8(fda))

    grid = np.concatenate([
        tensor_to_bgr_uint8(src),
        tensor_to_bgr_uint8(tgt),
        tensor_to_bgr_uint8(fda)
    ], axis=1)
    cv2.imwrite(str(out_dir / 'compare_side_by_side.png'), grid)

    print(f"Saved to {out_dir}")


if __name__ == '__main__':
    main()
    """
    python util/check_FDA.py --src /path/to/src.jpg --tgt /path/to/tgt.jpg --beta 0.09 --output_dir ./vis_out_fda
    """
