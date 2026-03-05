"""
COCO Polygon Preprocessor for the COCO-format CrowdAI Dataset

This script preprocesses the COCO-format CrowdAI dataset containing building polygon outlines.
The main goal is to localize the padding of each polygon, reducing the time needed to process data
during PolyR-CNN training.

It performs the following tasks:
1. For the training set, it pads each polygon to ensure a fixed number of vertices by uniformly sampling points along the polygon's outline and adds polygon corner classes.
2. For the test set, it only cleans noisy or invalid data.

The output is a modified COCO-format dataset specifically tailored for the CrowdAI dataset,
suitable for tasks that require fixed-vertex polygons.
"""
import numpy as np
import cv2
import math
import json
import os
import shapely.geometry
import time
from tqdm import tqdm
import argparse


def preprocess_annotation(json_path, save_path, num_corners, is_training=True):
    """
    Preprocess COCO-format annotations by cleaning noisy/invalid data and, for training data,
    padding polygon and adding ground truth corner classes.

    Original annotation format:
    - labels (dict) contains: {'info', 'categories', 'images', 'annotations'}
    - labels['annotations'] (list of dicts):
      - 'id', 'image_id', 'segmentation', 'area', 'bbox', 'category_id', 'iscrowd'
      - 'segmentation':
        - Vertices distributed clockwise with a random start point.
        - First and last vertices are identical.
        - Vertex coordinates are either int or float.
      - 'bbox':
        - wrong label.

    Preprocessed annotation format:
    - labels['annotations'] (list of dicts):
      - Additional key: 'cor_cls_poly' (added only for training sets)
      - 'segmentation':
        - Remove redundant vertices (closer than 0.1).
        - Simplify polygons (tolerance = 0.01).
        - Remove polygons with fewer than 3 vertices or smaller than 2 pixels².
        - Reorder vertices to start from the topmost point.
        - Clip coordinates to a valid range.
        - Uniformly sample vertices to pad the polygon (only for training sets).
      - 'bbox':
        - Enlarged by 20% around the polygon for ground truth bounding box.

    Args:
        json_path (str): Path to the original annotation file.
        save_path (str): Path to save the preprocessed annotation file.
        num_corners (int): Number of vertices to sample from the polygon.
        is_training (bool): If True, applies polygon padding and adds ground truth corner classes (for training).
                            If False, only cleans the data (for test sets).
    """
    print(f"Loading annotation from {json_path}...")
    with open(json_path, "r") as f:
        labels = json.load(f)
    print("Annotation loaded.")

    # Ensure 'info' key exists, which is required by COCO API
    if 'info' not in labels:
        labels['info'] = {
            "description": "WHU-Mix Dataset",
            "year": 2025,
            "contributor": "N/A",
            "date_created": "2025/10/13"
        }

    # Create a mapping from image_id to image shape
    image_id_to_shape = {
        img_info['id']: (img_info['height'], img_info['width'])
        for img_info in labels['images']
    }

    annotations = labels["annotations"]
    indices = []  # To store indices of invalid polygons

    # Process each annotation with a progress bar
    for i, anno in enumerate(tqdm(annotations, desc="Processing annotations")):
        # Get image dimensions
        image_id = anno['image_id']
        if image_id not in image_id_to_shape:
            indices.append(i)
            continue
        height, width = image_id_to_shape[image_id]

        # Get ground truth polygon
        gt_pts = anno["segmentation"][0]  # List [x1, y1, ..., xn, yn], float (mostly) or int

        # Get ground truth bounding box
        gt_bbox = get_gt_bboxes(gt_pts, width, height)  # [x1, y1, w, h]

        # Remove redundant vertices (closer than 0.1)
        gt_pts = np.array(gt_pts).reshape((-1, 2))  # numpy array, shape (N, 2), float64
        gt_pts = remove_doubles(gt_pts, epsilon=0.1)  # shape (N, 2), float64

        # Set area threshold depending on whether it's training or test set
        min_area = 2 if is_training else 10

        # Remove polygons with fewer than 3 vertices or an area smaller than the threshold
        if gt_pts.shape[0] < 3 or shapely.geometry.Polygon(gt_pts).area < min_area:
            indices.append(i)
            continue

        # Simplify the polygon
        gt_pts = approximate_polygons(gt_pts, tolerance=0.01)  # shape (N, 2), float64

        # Remove polygons again if they have fewer than 3 vertices or an area smaller than the threshold
        if gt_pts.shape[0] < 3 or shapely.geometry.Polygon(gt_pts).area < min_area:
            indices.append(i)
            continue

        # Reorder vertices to start from the top-left point (min x+y)
        # [WHU优化] 解决平顶建筑起始点跳变问题
        gt_pts = np.array(gt_pts).reshape((-1, 2))
        sum_xy = gt_pts[:, 0] + gt_pts[:, 1] # 计算 x + y
        ind = np.argmin(sum_xy)              # 找到 x+y 最小的索引
        gt_pts = np.concatenate((gt_pts[ind:], gt_pts[:ind]), axis=0)

        # Clip the polygon vertices to a valid range
        gt_pts = gt_pts.flatten()
        gt_pts[0::2] = np.clip(gt_pts[0::2], 0.0, width - 1e-4)
        gt_pts[1::2] = np.clip(gt_pts[1::2], 0.0, height - 1e-4)

        # If training, apply polygon padding and generate corner classification
        if is_training:
            # result = uniform_sampling(gt_pts, num_corners, width, height)
            result = uniform_sampling_vectorized(gt_pts, num_corners)
            if result[0] is None:
                indices.append(i)
                continue
            gt_pts, gt_cor_cls = result  # (num_corners * 2), (num_corners,), int32
            annotations[i]["cor_cls_poly"] = [int(c) for c in gt_cor_cls]  # Add polygon corner classes

        # Otherwise, just round and convert the polygon points to int (for test set)
        else:
            gt_pts = np.round(gt_pts).astype(np.int32)

        # Update annotation with new segmentation and bbox
        annotations[i]["segmentation"] = [[int(x) for x in gt_pts]]
        annotations[i]["bbox"] = gt_bbox

    # Remove invalid polygons
    indices = sorted(indices)
    for i in reversed(indices):
        annotations.pop(i)

    print(f"Processing complete. Saving to {save_path}...")

    # Save the updated annotation file
    labels["annotations"] = annotations
    with open(save_path, 'w') as fp:
        json.dump(labels, fp)

    print("Annotation file saved.")


def uniform_sampling(gt_pts, num_corners, width, height):
    """
    Uniformly sample a fixed number of vertices (num_corners) from the polygon contour, ensuring all
    annotated points are included. Annotated points are marked as class 0, and sampled points as class 1.

    Args:
        gt_pts (numpy.ndarray): Array of shape (N * 2), float64 representing the polygon points.
        num_corners (int): Number of vertices to sample from the polygon.

    Returns:
        encoded_polygon.flatten() (numpy.ndarray): Flattened array of shape (num_corners * 2), int32, rounded.
        corner_label (numpy.ndarray): Array of shape (num_corners,), int32 with labels 0 for corners, 1 for sampled points.
    """
    # Initialize corner labels, defaulting to class 1 (sampled points)
    corner_label = np.ones((num_corners,), dtype=np.int32)

    # Create a binary mask for the polygon
    polygon = np.round(gt_pts).astype(np.int32)
    polygon = polygon.reshape((-1, 1, 2))
    img = np.zeros((height, width), dtype="uint8")
    img = cv2.polylines(img, [polygon], True, 255, 1)
    img = cv2.fillPoly(img, [polygon], 255)

    # Find the contour of the polygon
    contour, _ = cv2.findContours(img, cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    if not contour:
        # If no contour is found, it might be a degenerate polygon.
        # Return None to indicate failure.
        return None, None
    contour = contour[0]  # Take the first contour (most likely the only one)

    lc = contour.shape[0]

    # Sample or pad the polygon to ensure exactly num_corners vertices
    if lc >= num_corners:
        ind = np.linspace(start=0, stop=lc, num=num_corners, endpoint=False)
        ind = np.round(ind).astype(np.int32)
        encoded_polygon = contour[ind].reshape((-1, 2))
    else:
        contour = contour.reshape((-1, 2))
        contour = [list(x) for x in contour]
        encoded_polygon = contour + [contour[-1]] * (num_corners - lc)  # Pad with the last vertex
        encoded_polygon = np.array(encoded_polygon).reshape((-1, 2))

    # Replace uniform sampled points with annotated points where necessary
    polygon = polygon.reshape((-1, 2))
    for i, x in enumerate(polygon):
        dists = [np.sqrt(np.sum((x - y) ** 2)) for y in encoded_polygon]  # Compute distances
        min_dist_idx = np.argmin(dists)  # Get the index of the closest sampled point
        if dists[min_dist_idx] == 0:
            corner_label[min_dist_idx] = 0  # Annotated point already in the sampled points, mark as class 0
        else:
            encoded_polygon[min_dist_idx] = x  # Replace sampled point with the annotated point
            corner_label[min_dist_idx] = 0  # Mark as class 0

    return encoded_polygon.flatten(), corner_label

def uniform_sampling_vectorized(gt_pts, num_corners):
    """
    矢量化均匀采样，并进行角点对齐。
    适配 engine.py 的在线训练需求。
    
    Args:
        gt_pts: list or np.ndarray, shape (N, 2) or (N*2,). 
                The raw polygon coordinates.
        num_corners: int. Target number of points (e.g., 64).
        
    Returns:
        sampled_flat: np.array (float32), shape (num_corners * 2,). Flattened coordinates.
        corner_label: np.array (int32), shape (num_corners,). 1 for corner, 0 for edge.
    """
    try:
        pts = np.array(gt_pts).reshape(-1, 2)
        
        # 1. 异常处理：点数太少
        if len(pts) < 3:
            # 退化情况：直接重复第一个点
            # 这种情况在训练中极少见，但为了防止报错崩掉训练进程
            return np.tile(pts[0], num_corners).flatten(), np.zeros(num_corners, dtype=np.int32)

        # 2. 确保闭合 (用于计算周长)
        if not np.array_equal(pts[0], pts[-1]):
            pts_closed = np.vstack([pts, pts[0]])
        else:
            pts_closed = pts

        # 3. 计算边长和周长
        diffs = np.diff(pts_closed, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        perimeter = np.sum(dists)
        
        if perimeter < 1e-6:
            return np.tile(pts[0], num_corners).flatten(), np.zeros(num_corners, dtype=np.int32)

        # 4. 累积距离 [0, d1, d1+d2, ..., perimeter]
        cum_dists = np.concatenate([[0], np.cumsum(dists)])

        # 5. 生成均匀采样位置
        # 注意：使用 endpoint=False，因为闭合多边形的起点和终点是同一个逻辑点
        sample_locs = np.linspace(0, perimeter, num_corners, endpoint=False)
        
        # 6. 线性插值计算采样点坐标
        # np.interp 需要 x 坐标单调递增，cum_dists 满足该条件
        new_x = np.interp(sample_locs, cum_dists, pts_closed[:, 0])
        new_y = np.interp(sample_locs, cum_dists, pts_closed[:, 1])
        sampled_pts = np.stack([new_x, new_y], axis=1) # Shape: (num_corners, 2)

        # --- 7. 角点对齐逻辑 (Corner Alignment) ---
        # 定义标签：默认为 0 (非角点/墙体点)
        corner_label = np.zeros((num_corners,), dtype=np.int32) 
        
        # 原始GT角点（去除重复尾点，如果输入未闭合则就是 pts）
        # 逻辑：真实的几何角点应该“吸附”到最近的采样点上
        if np.array_equal(pts[0], pts[-1]):
            orig_pts = pts[:-1]
        else:
            orig_pts = pts
        
        # 计算距离矩阵 (N_gt, N_sampled)
        # 利用广播机制计算所有对的距离
        dists_matrix = np.linalg.norm(orig_pts[:, None, :] - sampled_pts[None, :, :], axis=2)
        
        # 找到每个GT角点最近的采样点索引
        min_indices = np.argmin(dists_matrix, axis=1)
        
        # 执行替换和标记
        for gt_idx, sample_idx in enumerate(min_indices):
            # 强制将该采样点移动到真实的角点位置 (Pixel-perfect alignment)
            sampled_pts[sample_idx] = orig_pts[gt_idx] 
            # 标记为角点 (Class 1)
            corner_label[sample_idx] = 1 

        return sampled_pts.flatten(), corner_label

    except Exception as e:
        print(f"Error in uniform_sampling: {e}")
        # Fallback
        if len(gt_pts) > 0:
            p0 = np.array(gt_pts).reshape(-1, 2)[0]
            return np.tile(p0, num_corners).flatten(), np.zeros(num_corners, dtype=np.int32)
        return None, None


def get_gt_bboxes(gt_pts, width, height):
    """
    计算紧致的 Bounding Box，不进行外扩，适配 Deformable DETR。
    """
    gt_pts = np.array(gt_pts).reshape((-1, 2))
    x = gt_pts[:, 0]
    y = gt_pts[:, 1]

    xmin = np.min(x)
    ymin = np.min(y)
    xmax = np.max(x)
    ymax = np.max(y)
    
    # 稍微做一点点数值保护，防止宽高为0
    xmin = max(xmin, 0.0)
    ymin = max(ymin, 0.0)
    xmax = min(xmax, width - 1e-4)
    ymax = min(ymax, height - 1e-4)

    w = xmax - xmin
    h = ymax - ymin
    
    # 确保最小宽高
    if w < 1e-3: w = 1e-3
    if h < 1e-3: h = 1e-3

    return [float(xmin), float(ymin), float(w), float(h)]


def remove_doubles(vertices, epsilon=0.1):
    """
    Remove redundant vertices that are closer than a specified distance (epsilon).

    Args:
        vertices (numpy.ndarray): Array of shape (N, 2) representing the polygon vertices.
        epsilon (float, optional): Minimum distance between consecutive vertices to be considered non-redundant. Defaults to 0.1.
    """
    dists = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=-1)
    # np.roll(vertices, -1, axis=0): [[x_2, y_2], [x_3, y_3], ... , [x_n, y_n], [x_1, y_1]]
    # dists: [dist(v_2, v_1), dist(v_3, v_2), ... , dist(v_n, v_{n-1}), dist(v_1, v_n)]
    new_vertices = vertices[epsilon < dists]
    # If dist(v_{k}, v_{k-1}) < epsilon, remove v_{k-1}
    # If dist(v_{1}, v_{n}) < epsilon, remove v_{n}
    return new_vertices


def approximate_polygons(polygon, tolerance=0.01):
    """
    Approximate a polygonal chain with the specified tolerance using the Douglas-Peucker algorithm.

    Args:
        polygon (numpy.ndarray): Array of shape (N, 2) representing the polygon vertices.
        tolerance (float, optional): Maximum allowed distance between the original and simplified polygons. Defaults to 0.01.
    """
    from skimage.measure import approximate_polygon
    return approximate_polygon(polygon, tolerance)


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Preprocess COCO-format annotations for polygon padding and cleaning.")
    parser.add_argument('--json_path', required=True, help="Path to the input annotation file (JSON format).")
    parser.add_argument('--save_path', required=True, help="Path to save the preprocessed annotation file.")
    parser.add_argument('--is_training', type=bool, default=True,
                        help="Whether the dataset is for training (default: True).")
    parser.add_argument('--num_corners', type=int, required=True, help="Number of vertices to sample from the polygon.")

    return parser.parse_args()


if __name__ == '__main__':
    """"
    使用下面的指令运行脚本：
    python preprocess_annotation.py --json_path path/to/original/annotation.json \
                                    --save_path path/to/save/preprocessed/annotation.json \
                                    --is_training True \
                                    --num_corners 64
    """
    args = parse_args()
    preprocess_annotation(
        json_path=args.json_path,
        save_path=args.save_path,
        num_corners=args.num_corners,
        is_training=args.is_training
    )
