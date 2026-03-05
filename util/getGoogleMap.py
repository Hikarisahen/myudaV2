import math
import requests
from PIL import Image
from io import BytesIO
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import random

class DatasetBuilder:
    def __init__(self, output_dir="google_maps_uda_dataset", tile_size=256):
        self.output_dir = output_dir
        self.tile_size = tile_size
        self.servers = ['0', '1', '2', '3']
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.google.com/maps',
        }
        
        # 创建目标域无标签数据存放目录
        self.img_dir = os.path.join(self.output_dir, "unlabeled_images")
        os.makedirs(self.img_dir, exist_ok=True)
        self.image_counter = 1

    def _lon_to_tile_x(self, lon, zoom):
        return (lon + 180.0) / 360.0 * (2 ** zoom)

    def _lat_to_tile_y(self, lat, zoom):
        lat_rad = math.radians(lat)
        return (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * (2 ** zoom)

    def download_tile(self, x, y, z):
        """下载单张 256x256 瓦片"""
        server = random.choice(self.servers)
        url = f'http://mt{server}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                # 简单校验：过滤掉纯色无效图片（例如海面或未覆盖区域）
                extrema = img.convert('L').getextrema()
                if extrema[0] != extrema[1]:
                    return img
        except Exception:
            pass
        return None

    def create_512_patch(self, start_x, start_y, zoom):
        """下载 2x2 的瓦片阵列，拼接成 512x512 的图像用于模型输入"""
        patch = Image.new('RGB', (self.tile_size * 2, self.tile_size * 2))
        valid_tiles = 0

        # 获取左上、右上、左下、右下四个瓦片
        for i in range(2):
            for j in range(2):
                img = self.download_tile(start_x + i, start_y + j, zoom)
                if img:
                    patch.paste(img, (i * self.tile_size, j * self.tile_size))
                    valid_tiles += 1
                time.sleep(random.uniform(0.1, 0.3)) # 随机休眠防封锁

        # 只有当 4 张瓦片都成功下载时，才认为这个 512x512 的 patch 是完整的
        if valid_tiles == 4:
            return patch
        return None

    def build_from_bbox(self, region_name, min_lon, min_lat, max_lon, max_lat, zoom=19):
        """遍历指定边界框，按步长为 2 提取 512x512 的图像块"""
        print(f"\n开始采集区域: {region_name}")
        min_x = int(math.floor(self._lon_to_tile_x(min_lon, zoom)))
        max_x = int(math.floor(self._lon_to_tile_x(max_lon, zoom)))
        min_y = int(math.floor(self._lat_to_tile_y(max_lat, zoom)))
        max_y = int(math.floor(self._lat_to_tile_y(min_lat, zoom)))

        # 步长设为 2，因为每次抓取 2x2 瓦片 (512x512)
        x_coords = list(range(min_x, max_x, 2))
        y_coords = list(range(min_y, max_y, 2))
        total_patches = len(x_coords) * len(y_coords)
        
        print(f"预计可生成 512x512 样本数: {total_patches}")

        with tqdm(total=total_patches, desc=f"构建 {region_name} 数据集") as pbar:
            for x in x_coords:
                for y in y_coords:
                    patch_img = self.create_512_patch(x, y, zoom)
                    if patch_img:
                        # 统一命名格式，例如: target_domain_000001.jpg
                        filename = f"target_domain_{self.image_counter:06d}.jpg"
                        filepath = os.path.join(self.img_dir, filename)
                        patch_img.save(filepath, "JPEG", quality=95)
                        self.image_counter += 1
                    pbar.update(1)

# ================= 使用示例 =================
if __name__ == '__main__':
    builder = DatasetBuilder(output_dir="UDA_GoogleMaps_Dataset")
    
    # 填入你想要采集的目标域边界框 (WGS84坐标系: 最小经度, 最小纬度, 最大经度, 最大纬度)
    # 建议选取不同城市、不同建筑密度的区域，以提升泛化能力
    regions = [
        {"name": "City_Residential", "bbox": (116.38, 39.90, 116.47, 39.97)},   # 面积放大 ~9x
        {"name": "Suburban_Industrial", "bbox": (116.46, 39.76, 116.56, 39.86)},# 同样放大
        {"name": "Coastal_Port", "bbox": (121.58, 29.80, 121.72, 29.92)},        # 新增区域
        {"name": "NewTown", "bbox": (113.90, 22.45, 114.05, 22.60)},             # 新增区域
    ]
    
    # zoom=19 是对齐 crowdAI 0.3m 分辨率的最佳设定
    for region in regions:
        builder.build_from_bbox(
            region_name=region["name"],
            min_lon=region["bbox"][0],
            min_lat=region["bbox"][1],
            max_lon=region["bbox"][2],
            max_lat=region["bbox"][3],
            zoom=19 
        )
        
    print(f"\n✅ 数据集构建完成！共采集 {builder.image_counter - 1} 张有效图像。")
    print(f"数据存放路径: {os.path.abspath(builder.img_dir)}")