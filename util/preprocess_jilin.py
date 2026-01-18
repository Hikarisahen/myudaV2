import cv2
import os
import math
import glob
import numpy as np
import rasterio
from rasterio.enums import Resampling

# ================= 配置区域 =================
INPUT_DIR = "/data/zfx/datasets/Jilin-1" 
OUTPUT_DIR = "/data/zfx/datasets/Jilin-1/train"
TARGET_SCALE = 2.5  # 0.75m -> 0.3m
CROP_SIZE = 320     
OVERLAP = 100       
# ===========================================

def linear_stretch(img, percent=2):
    """
    对卫星图像进行 2% - 98% 线性拉伸，将 uint16 转为 uint8。
    这是卫星图像预处理的标准操作，能显著增强对比度。
    """
    # 确保是 float 计算
    img_float = img.astype(np.float32)
    
    # 计算截断阈值
    lower = np.percentile(img_float, percent)
    upper = np.percentile(img_float, 100 - percent)
    
    # 线性拉伸公式: (x - min) / (max - min) * 255
    img_stretched = (img_float - lower) / (upper - lower) * 255.0
    
    # 截断到 0-255 并转为 uint8
    img_stretched = np.clip(img_stretched, 0, 255).astype(np.uint8)
    return img_stretched

def preprocess_jilin1_image(large_img_path, output_dir):
    filename = os.path.basename(large_img_path)
    base_name = os.path.splitext(filename)[0]
    
    # === 修改 1: 使用 rasterio 读取 (支持多波段/16位) ===
    try:
        with rasterio.open(large_img_path) as src:
            print(f"正在处理 {filename}: 原始尺寸 {src.width}x{src.height}, 波段数={src.count}, 类型={src.dtypes[0]}")
            
            # 读取前3个波段 (通常是 RGB)
            # 注意: rasterio 读出来是 (C, H, W)，OpenCV 需要 (H, W, C)
            img = src.read([1, 2, 3]) # 读取 R, G, B
            img = np.transpose(img, (1, 2, 0)) # 转为 HWC
            
            # 如果是 BGR 顺序 (有些卫星是 BGR)，可能需要 img = img[..., ::-1]
            # 吉林一号通常是 RGB，OpenCV 保存需要 BGR，所以最后 cv2.imwrite 前要转一下，或者这里先处理
            # 暂时假设读取顺序正确，后续可视化如果颜色反了再调整
            
    except Exception as e:
        print(f"[Error] 无法读取 {large_img_path}: {e}")
        return

    # === 修改 2: 线性拉伸 (16bit -> 8bit) ===
    if img.dtype == np.uint16:
        print("  -> 检测到 16位 图像，正在执行 2% 线性拉伸转 8位...")
        img = linear_stretch(img)
    else:
        print("  -> 图像已是 8位，跳过拉伸。")

    h, w, c = img.shape
    
    # === 修改 3: 放大分辨率 ===
    new_w = int(w * TARGET_SCALE)
    new_h = int(h * TARGET_SCALE)
    print(f"  -> 正在放大 2.5倍 至 {new_w}x{new_h} (GSD=0.3m)...")
    
    # OpenCV 处理 uint8 很快
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # === 修改 4: 裁剪 ===
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    stride = CROP_SIZE - OVERLAP
    rows = math.ceil((new_h - CROP_SIZE) / stride) + 1
    cols = math.ceil((new_w - CROP_SIZE) / stride) + 1
    
    count = 0
    save_count = 0
    
    # 只需要 OpenCV 保存 (需要 BGR 格式)
    # Rasterio 读进来是 RGB，OpenCV write 需要 BGR
    img_resized_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

    for r in range(rows):
        for c in range(cols):
            y_start = int(r * stride)
            x_start = int(c * stride)
            
            y_end = min(y_start + CROP_SIZE, new_h)
            x_end = min(x_start + CROP_SIZE, new_w)
            
            # 边界修正
            if y_end - y_start < CROP_SIZE:
                y_start = max(0, y_end - CROP_SIZE)
            if x_end - x_start < CROP_SIZE:
                x_start = max(0, x_end - CROP_SIZE)
            
            crop = img_resized_bgr[y_start:y_start+CROP_SIZE, x_start:x_start+CROP_SIZE]
            
            # 过滤全黑或无效图片 (阈值设低一点，防止误杀)
            # 有了拉伸后，有效图像的 mean 不会很低
            if crop.mean() < 5: 
                continue
                
            save_name = os.path.join(output_dir, f"{base_name}_crop_{r}_{c}.jpg")
            cv2.imwrite(save_name, crop)
            save_count += 1
            
    print(f"  -> 完成！生成了 {save_count} 张切片。")

if __name__ == "__main__":
    tif_files = glob.glob(os.path.join(INPUT_DIR, "*.TIF")) + \
                glob.glob(os.path.join(INPUT_DIR, "*.tif"))
    
    print(f"找到 {len(tif_files)} 个大图文件。")
    
    for tif_file in tif_files:
        preprocess_jilin1_image(tif_file, OUTPUT_DIR)