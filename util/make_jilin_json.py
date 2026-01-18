import json
import os
import glob
from PIL import Image
from tqdm import tqdm

# ================= 配置区域 =================
# 必须与你报错信息中的路径完全一致
IMAGE_DIR = "/data/zfx/datasets/Jilin-1/train"
JSON_OUTPUT_PATH = "/data/zfx/datasets/Jilin-1/train/annotation_detr.json"
# ===========================================

def generate_json():
    print(f"正在扫描目录: {IMAGE_DIR} ...")
    
    # 1. 扫描所有 jpg 图片
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: 找不到目录 {IMAGE_DIR}")
        return

    image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    image_paths.sort() # 排序保证顺序一致
    
    if len(image_paths) == 0:
        print("Error: 目录下没有找到 .jpg 图片！请检查预处理步骤是否完成。")
        return

    images = []
    
    print(f"找到 {len(image_paths)} 张图片，正在生成 JSON 索引...")
    
    # 2. 遍历图片生成 info
    for i, img_path in enumerate(tqdm(image_paths)):
        filename = os.path.basename(img_path)
        
        # 为了速度，直接硬编码宽高（我们知道切片一定是 320x320）
        # 如果有边缘切片可能小于 320，建议读取真实尺寸，虽然慢一点但不出错
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except:
            # 如果读图失败，默认 320
            w, h = 320, 320
            
        images.append({
            "file_name": filename,
            "height": h,
            "width": w,
            "id": i  # 直接用索引作为 image_id
        })
        
    # 3. 构建 Categories (伪造一个类别，这就够了)
    categories = [
        {"id": 1, "name": "building", "supercategory": "building"}
    ]
    
    # 4. 构建最终字典 (annotations 为空)
    coco_output = {
        "images": images,
        "annotations": [], # 关键：这里留空！Teacher 会在训练时填补这里
        "categories": categories
    }
    
    # 5. 保存
    print(f"正在写入文件: {JSON_OUTPUT_PATH}")
    with open(JSON_OUTPUT_PATH, 'w') as f:
        json.dump(coco_output, f)
    
    print("✅ 成功！索引文件已生成。")

if __name__ == "__main__":
    generate_json()