import rasterio
import sys
import os

def get_gsd(image_path):
    """
    尝试从 GeoTIFF 元数据中读取 GSD。
    """
    if not os.path.exists(image_path):
        print(f"[Error] File not found: {image_path}")
        return None

    try:
        with rasterio.open(image_path) as src:
            # src.res 返回 (x_resolution, y_resolution)
            res_x, res_y = src.res
            
            # 也就是 transform 中的参数
            transform = src.transform
            
            print(f"--- File: {os.path.basename(image_path)} ---")
            print(f"Driver: {src.driver}")
            print(f"Size: {src.width} x {src.height}")
            print(f"Coordinate System (CRS): {src.crs}")
            
            # 判断单位
            if src.crs and src.crs.is_geographic:
                print("Warning: Coordinate system is Geographic (Lat/Lon). Resolution is in degrees, not meters!")
                print(f"Resolution: {res_x:.8f} degrees/pixel")
                # 粗略估算：在赤道附近，0.00001 度 ≈ 1.11 米
                est_meters = res_x * 111320
                print(f"Estimated GSD: ~{est_meters:.2f} meters/pixel (Lat dependent)")
            else:
                # 投影坐标系 (Projected)，单位通常是米
                print(f"Resolution: {res_x:.4f} x {res_y:.4f} (usually meters)")
                if abs(res_x - res_y) > 0.01:
                    print("Note: Pixel is not square.")
                return res_x

    except Exception as e:
        print(f"[Info] Cannot read geospatial info from {os.path.basename(image_path)}.")
        print(f"Reason: {e}")
        # 对于普通的 JPG/PNG，没有地理信息
        return None

if __name__ == "__main__":
    # 在这里替换成你的图片路径
    # 1. 检查吉林一号大图 (通常是 .tif)
    jilin_path = "/data/zfx/datasets/Jilin-1/tile_0.TIF"
    
    # 2. 检查 CrowdAI 小图 (如果是 jpg，通常读不到，需要查文档)
    crowdai_path = "/data/zfx/datasets/CrowdAI/train/images/000000000042.jpg"
    
    print("Checking Jilin-1...")
    get_gsd(jilin_path)
    
    print("\nChecking CrowdAI...")
    get_gsd(crowdai_path)