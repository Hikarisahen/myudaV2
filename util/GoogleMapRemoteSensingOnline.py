"""
遥感图像下载工具 - GeoTIFF 版本
支持输出带地理坐标的 GeoTIFF 格式
"""

import math
import random
import requests
from PIL import Image
from io import BytesIO
import os
# 强制使用当前 conda 环境的 PROJ/GDAL（在导入 rasterio 之前）
_cp = os.environ.get("CONDA_PREFIX")
if _cp:
    os.environ["PROJ_LIB"] = os.path.join(_cp, "Library", "share", "proj")
    os.environ["GDAL_DATA"] = os.path.join(_cp, "Library", "share", "gdal")
    try:
        os.add_dll_directory(os.path.join(_cp, "Library", "bin"))
    except Exception:
        pass

from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import numpy as np

# ===== 坐标系转换工具：BD-09 / GCJ-02 / WGS84 =====
# 说明：
#   - 百度 place/v3/region 返回 BD-09 坐标
#   - Google/Sentinel 以及本工具内部统一使用 WGS84 (EPSG:4326)
#   - 因此需要: BD-09 -> GCJ-02 -> WGS84

X_PI = math.pi * 3000.0 / 180.0
PI = math.pi
AXIS = 6378245.0  # a
EE = 0.00669342162296594323  # e^2


def _out_of_china(lon: float, lat: float) -> bool:
    return not (72.004 <= lon <= 137.8347 and 0.8293 <= lat <= 55.8271)


def _transform_lat(lon: float, lat: float) -> float:
    ret = -100.0 + 2.0 * lon + 3.0 * lat + 0.2 * lat * lat + 0.1 * lon * lat + 0.2 * math.sqrt(abs(lon))
    ret += (20.0 * math.sin(6.0 * lon * PI) + 20.0 * math.sin(2.0 * lon * PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * PI) + 40.0 * math.sin(lat / 3.0 * PI)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * PI) + 320 * math.sin(lat * PI / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lon(lon: float, lat: float) -> float:
    ret = 300.0 + lon + 2.0 * lat + 0.1 * lon * lon + 0.1 * lon * lat + 0.1 * math.sqrt(abs(lon))
    ret += (20.0 * math.sin(6.0 * lon * PI) + 20.0 * math.sin(2.0 * lon * PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lon * PI) + 40.0 * math.sin(lon / 3.0 * PI)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lon / 12.0 * PI) + 300.0 * math.sin(lon / 30.0 * PI)) * 2.0 / 3.0
    return ret


def gcj02_to_wgs84(lon: float, lat: float) -> Tuple[float, float]:
    """GCJ-02(火星坐标) -> WGS84"""
    if _out_of_china(lon, lat):
        return lon, lat
    d_lat = _transform_lat(lon - 105.0, lat - 35.0)
    d_lon = _transform_lon(lon - 105.0, lat - 35.0)
    rad_lat = lat / 180.0 * PI
    magic = math.sin(rad_lat)
    magic = 1 - EE * magic * magic
    sqrt_magic = math.sqrt(magic)
    d_lat = (d_lat * 180.0) / ((AXIS * (1 - EE)) / (magic * sqrt_magic) * PI)
    d_lon = (d_lon * 180.0) / (AXIS / sqrt_magic * math.cos(rad_lat) * PI)
    mg_lat = lat + d_lat
    mg_lon = lon + d_lon
    return lon - d_lon, lat - d_lat


def bd09_to_gcj02(bd_lon: float, bd_lat: float) -> Tuple[float, float]:
    """BD-09(百度) -> GCJ-02(火星坐标)"""
    x = bd_lon - 0.0065
    y = bd_lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * X_PI)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * X_PI)
    gcj_lon = z * math.cos(theta)
    gcj_lat = z * math.sin(theta)
    return gcj_lon, gcj_lat


try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    print(rasterio.__version__)
    print(os.environ.get('PROJ_LIB'))
    GEOTIFF_SUPPORT = True
except ImportError:
    GEOTIFF_SUPPORT = False
    print("⚠ 警告: 未安装 rasterio，无法生成 GeoTIFF")
    print("   请运行: pip install rasterio")


"""
使用百度地图 API 获取地点边界框
"""

def baidu_place_to_bbox(place_name: str,
                        api_key: str,
                        padding_km: float = 0.5) -> Optional[Tuple[float, float, float, float]]:
    """
    使用百度地图 API 将地点转换为左下角和右上角的边界框 (WGS84 经纬度)

        Args:
            place_name: 地点名称（如 "天安门", "朝阳公园"）
            api_key: 百度地图开发者 API Key
            padding_km: 边界留白（单位: 公里）

        Returns:
            (min_lon, min_lat, max_lon, max_lat) 或 None

        Example:
            bbox = baidu_place_to_bbox("天安门", api_key="your_baidu_api_key")
            print(bbox)  # (116.384, 39.897, 116.414, 39.918)
    """
    # 百度地图的地理编码 API 地址
    # url = "http://api.map.baidu.com/geocoding/v3/"
    # params = {
    #     "address": place_name,
    #     "output": "json",
    #     "ak": api_key,  # API Key
    # }

    # 地点检索V3
    url = "https://api.map.baidu.com/place/v3/region"
    params = {
        "query": place_name,
        "region": "中国",
        "ak": api_key,
    }
    

    # # 地址解析聚合(高级功能，需要认证后付费使用)
    # url = "https://api.map.baidu.com/address_analyzer/v2"
    # params = {
    #     "address": place_name,
    #     "ak": api_key,
    # }

    try:
        # 发起请求获取地理位置信息
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        print("="*50)
        print(f"data: {data}")
        print("="*50)

        # 检查响应状态
        if data.get("status") != 0:
            print(f"✗ API 调用失败: {data.get('message') or data.get('msg', '未知错误')}")
            return None

        # 对于 place/v3/region，结果在 result 是一个列表，取第一个 POI
        results = data.get("results") or data.get("result")
        if not results:
            print("✗ 未返回任何地点结果")
            return None

        first = results[0]
        location = first.get("location", {})
        bd_lat = location.get("lat")
        bd_lon = location.get("lng")

        if bd_lat is None or bd_lon is None:
            print("✗ 未能在第一个结果中找到有效坐标")
            return None

        print(f"✓ 使用第一个地点: {first.get('name', '')}")
        print(f"✓ 百度坐标系(BD-09) 中心点: ({bd_lon:.6f}, {bd_lat:.6f})")

        # 1) BD-09 -> GCJ-02
        gcj_lon, gcj_lat = bd09_to_gcj02(bd_lon, bd_lat)
        # 2) GCJ-02 -> WGS84 (EPSG:4326)
        wgs_lon, wgs_lat = gcj02_to_wgs84(gcj_lon, gcj_lat)

        print(f"✓ 转换后 WGS84(EPSG:4326) 中心点: ({wgs_lon:.6f}, {wgs_lat:.6f})")

        # 使用 WGS84 中心点计算边界框
        min_lon, min_lat, max_lon, max_lat = _calculate_bbox(wgs_lon, wgs_lat, padding_km)
        print(f"✓ 基于 WGS84 计算边界框: ({min_lon:.6f}, {min_lat:.6f}) 到 ({max_lon:.6f}, {max_lat:.6f})")

        return min_lon, min_lat, max_lon, max_lat

    except requests.exceptions.RequestException as e:
        print(f"⚠ 网络请求错误: {e}")
        return None
    except Exception as e:
        print(f"⚠ 未知错误: {e}")
        return None


def _calculate_bbox(center_lon: float, center_lat: float, padding_km: float) -> Tuple[float, float, float, float]:
    """
    根据中心点和留白计算边界框

    Args:
        center_lon: 中心点经度
        center_lat: 中心点纬度
        padding_km: 要在中心点周围留白的距离 (单位: 公里)

    Returns:
        矩形边界框: (min_lon, min_lat, max_lon, max_lat)
    """
    # 经纬度到距离换算
    lat_offset = padding_km / 111.32  # 1 纬度度数 ≈ 111.32 公里
    lon_offset = padding_km / (111.32 * abs(math.cos(math.radians(center_lat))))  # 经度度数随纬度变化

    min_lon = center_lon - lon_offset
    max_lon = center_lon + lon_offset
    min_lat = center_lat - lat_offset
    max_lat = center_lat + lat_offset

    return min_lon, min_lat, max_lon, max_lat


class RemoteSensingDownloader:
    """遥感影像下载器 - 支持 GeoTIFF"""
    
    TILE_SOURCES = {
        'google_satellite': {
            'url': 'http://mt{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            'name': 'Google 卫星影像',
            'max_zoom': 20,
            'servers': ['0', '1', '2', '3'],
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://www.google.com/maps',
            }
        },
        'google_terrain': {
            'url': 'http://mt{s}.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
            'name': 'Google 地形图',
            'max_zoom': 20,
            'servers': ['0', '1', '2', '3'],
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://www.google.com/maps',
            }
        },
        'google_hybrid': {
            'url': 'http://mt{s}.google. com/vt/lyrs=y&x={x}&y={y}&z={z}',
            'name': 'Google 混合地图',
            'max_zoom': 20,
            'servers': ['0', '1', '2', '3'],
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://www.google.com/maps',
            }
        }
    }
    
    def __init__(self, source: str = 'google_satellite', debug: bool = False):
        """
        初始化下载器
        
        Args:
            source: 瓦片源名称
            debug: 调试模式
        """
        if source not in self.TILE_SOURCES:
            raise ValueError(f"不支持的数据源: {source}")
        
        self.source_config = self.TILE_SOURCES[source]
        self. tile_url_template = self.source_config['url']
        self.servers = self.source_config. get('servers', ['1'])
        self.tile_size = 256
        self.headers = self.source_config.get('headers', {})
        self.debug = debug
        
        self.stats = {'success': 0, 'failed': 0, 'total': 0}
        self.server_index = 0
    
    def _get_next_server(self) -> str:
        """获取下一个服务器"""
        server = self.servers[self.server_index % len(self.servers)]
        self.server_index += 1
        return server
    
    def _lon_to_tile_x(self, lon: float, zoom: int) -> float:
        """经度转瓦片 X 坐标"""
        return (lon + 180.0) / 360.0 * (2 ** zoom)
    
    def _lat_to_tile_y(self, lat: float, zoom: int) -> float:
        """纬度转瓦片 Y 坐标（Web Mercator）"""
        lat_rad = math.radians(lat)
        return (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * (2 ** zoom)

    def _tile_to_lon(self, x: float, zoom: int) -> float:
        """瓦片 X 坐标转经度"""
        return x / (2 ** zoom) * 360.0 - 180.0
    
    def _tile_to_lat(self, y: float, zoom: int) -> float:
        """瓦片 Y 坐标转纬度"""
        n = math.pi - 2.0 * math.pi * y / (2 ** zoom)
        return math.degrees(math.atan(math.sinh(n)))
    
    def get_tile_range(self, 
                       min_lon: float, 
                       min_lat: float, 
                       max_lon: float, 
                       max_lat: float, 
                       zoom: int) -> Tuple[int, int, int, int]:
        """计算瓦片范围"""
        min_x = int(math.floor(self._lon_to_tile_x(min_lon, zoom)))
        max_x = int(math.floor(self._lon_to_tile_x(max_lon, zoom)))
        min_y = int(math.floor(self._lat_to_tile_y(max_lat, zoom)))
        max_y = int(math.floor(self._lat_to_tile_y(min_lat, zoom)))
        
        return min_x, min_y, max_x, max_y
    
    def download_tile(self, x: int, y: int, z: int, retry: int = 3) -> Optional[Image.Image]:
        """下载单个瓦片"""
        for attempt in range(retry):
            try:
                server = self._get_next_server()
                url = self.tile_url_template.format(s=server, x=x, y=y, z=z)
                
                if self.debug and attempt == 0:
                    print(f"下载: ({x}, {y}, {z})")
                
                response = requests.get(url, headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    content_type = response.headers.get('Content-Type', '')
                    if 'image' in content_type or len(response.content) > 1000:
                        img = Image.open(BytesIO(response.content))
                        
                        # 验证不是单色图像
                        extrema = img.convert('L').getextrema()
                        if extrema[0] != extrema[1]:
                            self.stats['success'] += 1
                            return img
                
                elif response.status_code == 404:
                    return None
                    
            except Exception as e:
                if self.debug:
                    print(f"  错误: {e}")
                time.sleep(0.5)
        
        self.stats['failed'] += 1
        return None
    
    def download_region(self,
                       min_lon: float,
                       min_lat: float,
                       max_lon: float,
                       max_lat: float,
                       zoom: int,
                       output_path: str,
                       max_workers: int = 5,
                       output_format: str = 'auto') -> Optional[str]:
        """
        下载指定区域的遥感影像
        
        Args:
            min_lon: 最小经度
            min_lat: 最小纬度
            max_lon: 最大经度
            max_lat: 最大纬度
            zoom: 缩放级别
            output_path: 输出路径
            max_workers: 并发数
            output_format: 输出格式 ('auto', 'geotiff', 'jpg', 'png')
                          'auto' - 根据文件扩展名自动判断
                          'geotiff' - 强制输出 GeoTIFF
            
        Returns:
            输出文件路径
        """
        # 参数验证
        if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
            raise ValueError("经度必须在 -180 到 180 之间")
        if not (-85.0511 <= min_lat <= 85.0511 and -85.0511 <= max_lat <= 85.0511):
            raise ValueError("纬度必须在 -85.0511 到 85.0511 之间")
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError("最小坐标必须小于最大坐标")
        
        # 判断输出格式
        ext = os.path.splitext(output_path)[1].lower()
        if output_format == 'auto':
            if ext in ['.tif', '.tiff']:
                output_format = 'geotiff'
            elif ext in ['.jpg', '. jpeg']:
                output_format = 'jpg'
            elif ext == '.png':
                output_format = 'png'
            else:
                output_format = 'jpg'
        
        # 检查 GeoTIFF 支持
        if output_format == 'geotiff' and not GEOTIFF_SUPPORT:
            print("⚠ GeoTIFF 不支持，自动切换为 JPG 格式")
            output_format = 'jpg'
            output_path = os.path.splitext(output_path)[0] + '.jpg'
        
        max_zoom = self.source_config['max_zoom']
        if zoom > max_zoom:
            print(f"⚠ 缩放级别调整: {zoom} -> {max_zoom}")
            zoom = max_zoom
        
        print(f"\n{'='*70}")
        print(f"🛰  遥感影像下载任务")
        print(f"{'='*70}")
        print(f"数据源:     {self.source_config['name']}")
        print(f"区域范围:   经度 [{min_lon:.6f}, {max_lon:.6f}]")
        print(f"            纬度 [{min_lat:.6f}, {max_lat:.6f}]")
        print(f"缩放级别:   {zoom}")
        print(f"输出格式:   {output_format. upper()}")
        
        # 计算瓦片范围
        min_x, min_y, max_x, max_y = self.get_tile_range(
            min_lon, min_lat, max_lon, max_lat, zoom
        )
        
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        total_tiles = width * height
        
        # 计算实际覆盖的地理范围（基于瓦片边界）
        actual_min_lon = self._tile_to_lon(min_x, zoom)
        actual_max_lon = self._tile_to_lon(max_x + 1, zoom)
        actual_min_lat = self._tile_to_lat(max_y + 1, zoom)
        actual_max_lat = self._tile_to_lat(min_y, zoom)
        
        print(f"瓦片网格:   {width} × {height} = {total_tiles} 张")
        print(f"图像尺寸:   {width * self.tile_size} × {height * self.tile_size} 像素")
        print(f"实际范围:   经度 [{actual_min_lon:.6f}, {actual_max_lon:.6f}]")
        print(f"            纬度 [{actual_min_lat:.6f}, {actual_max_lat:.6f}]")
        
        # if total_tiles > 500:
        #     estimated_mb = total_tiles * 0.015
        #     print(f"⚠ 预计下载: {estimated_mb:.1f} MB")
        #     response = input("是否继续？(y/n): ")
        #     if response.lower() != 'y':
        #         return None
        
        # 重置统计
        self.stats = {'success': 0, 'failed': 0, 'total': total_tiles}
        
        # 创建画布
        output_image = Image.new('RGB', 
                                (width * self.tile_size, height * self.tile_size),
                                color=(255, 255, 255))
        
        print(f"\n开始下载 (并发数: {max_workers})...")
        
        # 多线程下载
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            for i in range(width):
                for j in range(height):
                    x = min_x + i
                    y = min_y + j
                    future = executor.submit(self.download_tile, x, y, zoom)
                    futures[future] = (i, j)
            
            with tqdm(total=total_tiles, desc="📥 下载进度", unit="tile") as pbar:
                for future in as_completed(futures):
                    i, j = futures[future]
                    tile = future.result()
                    
                    if tile is not None:
                        output_image.paste(tile, (i * self. tile_size, j * self. tile_size))
                    
                    pbar.update(1)
        
        # 保存文件
        print(f"\n💾 保存文件...")
        os.makedirs(os.path. dirname(output_path) or '.', exist_ok=True)
        
        if output_format == 'geotiff':
            # 保存为 GeoTIFF
            self._save_as_geotiff(
                output_image, 
                output_path,
                actual_min_lon, 
                actual_min_lat, 
                actual_max_lon, 
                actual_max_lat
            )
        elif output_format == 'png':
            output_image.save(output_path, 'PNG', optimize=True)
        else:  # jpg
            output_image. save(output_path, 'JPEG', quality=95, optimize=True)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        success_rate = (self.stats['success'] / self.stats['total'] * 100) if self.stats['total'] > 0 else 0
        
        # 结果报告
        print(f"\n{'='*70}")
        print(f"✅ 下载完成！")
        print(f"{'='*70}")
        print(f"输出文件:   {os.path.abspath(output_path)}")
        print(f"图像尺寸:   {output_image.size[0]} × {output_image.size[1]} 像素")
        print(f"文件大小:   {file_size:.2f} MB")
        print(f"成功率:     {self.stats['success']}/{self.stats['total']} ({success_rate:.1f}%)")
        
        if output_format == 'geotiff':
            print(f"坐标系统:   EPSG:4326 (WGS84)")
            print(f"地理范围:   {actual_min_lon:.6f}, {actual_min_lat:.6f}, {actual_max_lon:.6f}, {actual_max_lat:.6f}")
        
        if success_rate < 80:
            print(f"\n⚠ 成功率较低，建议降低 zoom 或缩小范围")
        
        return output_path
    
    def _save_as_geotiff(self, 
                        image: Image.Image, 
                        output_path: str,
                        min_lon: float, 
                        min_lat: float, 
                        max_lon: float, 
                        max_lat: float):
        """
        保存为 GeoTIFF 格式
        
        Args:
            image: PIL Image 对象
            output_path: 输出路径
            min_lon: 最小经度
            min_lat: 最小纬度
            max_lon: 最大经度
            max_lat: 最大纬度
        """
        if not GEOTIFF_SUPPORT:
            raise RuntimeError("未安装 rasterio，无法生成 GeoTIFF")
        
        # 转换为 numpy 数组
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # 创建仿射变换矩阵（从像素坐标到地理坐标）
        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

        # 优先使用 EPSG:4326，失败则回退到 WKT
        try:
            target_crs = CRS.from_epsg(4326)
        except Exception:
            target_crs = CRS.from_wkt(
                'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
            )

        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=3,
            dtype=img_array.dtype,
            crs=target_crs,
            transform=transform,
            compress='lzw',
            tiled=True,
            blockxsize=256,
            blockysize=256
        ) as dst:
            # 写入三个波段（R, G, B）
            for i in range(3):
                dst.write(img_array[:, :, i], i + 1)
            
            # 添加元数据
            dst.update_tags(
                source=self.source_config['name'],
                description='Downloaded by RemoteSensingDownloader',
                creation_date=time.strftime('%Y-%m-%d %H:%M:%S')
            )

# ======== 封装为可复用工具 =========
def getRemoteSensing(place_name: str, api_key: str = "vlefQXy9Grrg5BkorLW2Yyx0n6JiFTvH") -> Optional[str]:
    '''
    根据地点名称获取遥感影像并保存为 GeoTIFF 文件(google 卫星影像)
    param :
        place_name: 地点名称
        api_key: 百度地图 API Key
    return:
        输出文件路径
    '''
    downloader = RemoteSensingDownloader(source='google_satellite')
    min_lon, min_lat, max_lon, max_lat = baidu_place_to_bbox(place_name=place_name, api_key=api_key)
    random_numb = random.randint(10000,99999)
    downloader.download_region(
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        zoom=30,
        output_path=f'langgraph_agent/tools/RS_Online/output/{random_numb}.tif',
        max_workers=10,
        output_format='geotiff'
    )

    return f'langgraph_agent/tools/RS_Online/output/{random_numb}.tif'


def main():
    '''
        获取遥感影像
        downloader = RemoteSensingDownloader(source='google_satellite')

        获取地形图
        downloader = RemoteSensingDownloader(source='google_terrain')

        获取混合地图
        downloader = RemoteSensingDownloader(source='google_hybrid')
    '''
    print("="*70)
    print("遥感影像下载工具 - GeoTIFF 版本")
    print("="*70)
    
    downloader = RemoteSensingDownloader(source='google_satellite')
    place_name = "武汉市武昌区"
    min_lon, min_lat, max_lon, max_lat = baidu_place_to_bbox(place_name=place_name, api_key="vlefQXy9Grrg5BkorLW2Yyx0n6JiFTvH")

    print("="*70)
    print(f"下载区域{place_name}: ({min_lon}, {min_lat}, {max_lon}, {max_lat})")
    print("="*70)

    print("\n示例: 输出 GeoTIFF 格式")
    downloader.download_region(
        min_lon=min_lon,   # 西南角经度
        min_lat=min_lat,  # 西南角纬度
        max_lon=max_lon,  # 东北角经度
        max_lat=max_lat,  # 东北角纬度
        # zoom=30,  # 缩放级别,他越大细节越多,范围越小
        zoom=20,  # 缩放级别,他越大细节越多,范围越小
        output_path=f'langgraph_agent/tools/RS_Online/output/{place_name}.tif',  # 输出路径
        max_workers=10,  # 并发下载线程数
        output_format='geotiff'  # 指定格式，设置为auto则根据文件扩展名自动判断
    )


if __name__ == '__main__':
    main()