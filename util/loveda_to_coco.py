import os
import cv2
import json
import numpy as np
from tqdm import tqdm

def create_coco_dict():
    """初始化 COCO 格式的字典结构"""
    return {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "building", "supercategory": "building"}]
    }

def process_loveda_masks(mask_dir, output_json, image_width=1024, image_height=1024):
    coco_data = create_coco_dict()
    ann_id = 1
    
    # 获取所有 png 掩码文件
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    
    for image_id, mask_name in enumerate(tqdm(mask_files, desc="Processing Masks")):
        # 1. 注册图像信息
        # 注意：LoveDA 掩码文件名通常与原图文件名一致，这里直接使用掩码名作为 file_name
        image_info = {
            "id": image_id + 1,
            "width": image_width,
            "height": image_height,
            "file_name": mask_name  
        }
        coco_data["images"].append(image_info)
        
        # 2. 读取掩码并提取建筑物 (LoveDA 中建筑物类别像素值为 2)
        mask_path = os.path.join(mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 二值化：建筑物区域为 255，其他为 0
        binary = np.zeros_like(mask)
        binary[mask == 2] = 255
        
        if np.max(binary) == 0:
            continue # 如果这张图没有建筑物，跳过标注提取
            
        # 3. 形态学操作与分水岭算法分离粘连建筑物
        kernel = np.ones((3, 3), np.uint8)
        # 开运算去除小的噪点
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 确定背景区域（膨胀操作）
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # 确定前景区域（距离变换寻找建筑物中心）
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        # 这里的 0.3 是一个超参数，值越大分离得越碎，值越小粘连越多。根据效果可微调
        ret, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # 寻找未知区域（边界）
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # 标记连通域
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1 # 背景标记为 1
        markers[unknown == 255] = 0 # 未知区域标记为 0
        
        # 应用分水岭算法
        mask_3c = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR) # 分水岭需要三通道输入
        markers = cv2.watershed(mask_3c, markers)
        
        # 4. 提取实例轮廓并转为 COCO 格式
        # 忽略背景(1)和边界(-1)，提取每个独立的建筑物实例
        unique_markers = np.unique(markers)
        for marker in unique_markers:
            if marker == 1 or marker == -1:
                continue
                
            # 提取单个建筑物的掩码
            instance_mask = np.zeros_like(mask, dtype=np.uint8)
            instance_mask[markers == marker] = 255
            
            # 寻找轮廓
            contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
                
            # 获取最大轮廓（过滤掉可能的微小碎片）
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            
            # 过滤掉面积太小的噪点框（例如小于 10 个像素）
            if area < 10:
                continue
                
            # 获取边界框 [x, y, width, height]
            x, y, w, h = cv2.boundingRect(contour)
            
            # 将多边形轮廓展平为 COCO 需要的格式 [x1, y1, x2, y2, ...]
            segmentation = contour.flatten().tolist()
            
            # 确保多边形至少有 3 个点（6 个坐标）
            if len(segmentation) < 6:
                continue
                
            ann_info = {
                "id": ann_id,
                "image_id": image_id + 1,
                "category_id": 1, # 我们将建筑物统一定义为类别 1
                "segmentation": [segmentation],
                "bbox": [x, y, w, h],
                "area": area,
                "iscrowd": 0
            }
            coco_data["annotations"].append(ann_info)
            ann_id += 1

    # 5. 保存 JSON 文件
    print(f"Saving COCO annotations to {output_json}...")
    with open(output_json, 'w') as f:
        json.dump(coco_data, f)
    print(f"Done! Generated {ann_id - 1} bounding boxes.")

# ==========================================
# 运行配置区
# ==========================================
if __name__ == "__main__":
    # 替换为你本地 LoveDA 验证集掩码所在的路径
    LOVE_DA_VAL_MASKS_DIR = "./LoveDA/Val/Masks" 
    
    # 输出的 COCO json 文件名
    OUTPUT_COCO_JSON = "loveda_val_coco_format.json"
    
    # 确保目录存在的情况下运行
    if os.path.exists(LOVE_DA_VAL_MASKS_DIR):
        process_loveda_masks(LOVE_DA_VAL_MASKS_DIR, OUTPUT_COCO_JSON)
    else:
        print(f"Error: Mask directory '{LOVE_DA_VAL_MASKS_DIR}' not found.")