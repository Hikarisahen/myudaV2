# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math
from typing import Tuple, List

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
import copy
from torchvision.ops import sigmoid_focal_loss


def get_box_template(num_points=64, device='cpu'):
    """
    生成一个单位矩形模板 [0,1]x[0,1]，点按逆时针排列。
    """
    # 4条边，每条边 num_points // 4 个点
    n_side = num_points // 4
    
    # Top: (x, 0) x from 0 to 1
    top_x = torch.linspace(0, 1, n_side, device=device)
    top_y = torch.zeros(n_side, device=device)
    top = torch.stack([top_x, top_y], dim=1)
    
    # Right: (1, y) y from 0 to 1
    right_x = torch.ones(n_side, device=device)
    right_y = torch.linspace(0, 1, n_side, device=device)
    right = torch.stack([right_x, right_y], dim=1)
    
    # Bottom: (x, 1) x from 1 to 0
    bottom_x = torch.linspace(1, 0, n_side, device=device)
    bottom_y = torch.ones(n_side, device=device)
    bottom = torch.stack([bottom_x, bottom_y], dim=1)
    
    # Left: (0, y) y from 1 to 0
    left_x = torch.zeros(n_side, device=device)
    left_y = torch.linspace(1, 0, n_side, device=device)
    left = torch.stack([left_x, left_y], dim=1)
    
    # Concat: (N, 2)
    template = torch.cat([top, right, bottom, left], dim=0)
    
    # 如果 num_points 不能被 4 整除，补齐
    if template.shape[0] < num_points:
        diff = num_points - template.shape[0]
        last = template[-1:]
        template = torch.cat([template, last.repeat(diff, 1)], dim=0)
        
    return template

def get_geometric_features(poly):
    """
    计算多边形的几何特征：相对位移和夹角余弦。
    poly: (B, Q, N, 2)
    Returns: (B, Q, N, 5) -> [diff_prev(2), diff_next(2), angle_cos(1)]
    """
    # Roll to get neighbors (cyclic)
    # prev: i-1, next: i+1
    prev_poly = torch.roll(poly, shifts=1, dims=2)
    next_poly = torch.roll(poly, shifts=-1, dims=2)
    
    # Relative vectors
    # vector pointing from prev to current
    diff_prev = poly - prev_poly # (B, Q, N, 2)
    # vector pointing from current to next
    diff_next = next_poly - poly # (B, Q, N, 2)
    
    # Normalize for angle calculation
    # Add epsilon to avoid div by zero
    norm_prev = diff_prev.norm(dim=-1, keepdim=True) + 1e-6
    norm_next = diff_next.norm(dim=-1, keepdim=True) + 1e-6
    
    vec_prev = diff_prev / norm_prev
    vec_next = diff_next / norm_next
    
    # Cosine similarity: dot product
    # Dot product of incoming vector and outgoing vector
    # If straight line, vectors are same dir, cos = 1.
    # If 90 deg turn, cos = 0.
    angle_cos = (vec_prev * vec_next).sum(dim=-1, keepdim=True) # (B, Q, N, 1)
    
    return torch.cat([diff_prev, diff_next, angle_cos], dim=-1) # 5 dims

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, use_mae=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        # [新增] 初始化 MAE Decoder
        self.use_mae = use_mae
        if self.use_mae:
            # feature_size 取决于你的 Backbone 和输入图像尺寸
            # 这里的 (10, 10) 是基于 input size (320x320) / 32 的估算值
            # feature_size是backbone最后一层特征图的尺寸
            # 可以改成自适应，或者作为超参传入
            self.mae_decoder = AuxiliaryMAEDecoder(
                d_model=transformer.d_model, 
                feature_size=(10, 10) # 这里稍微给大一点或者根据实际情况调整
            )
            # 这里的 Feature Size 其实不影响 query_embed 的大小，只要足够覆盖最大特征图即可
            # 或者为了严谨，我们可以在 forward 里动态生成 query_embed (如果支持变长)
            # 但为了遵循 MRT 论文，使用固定 Query 也是可以的
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # [新增] 多边形预测配置
        self.num_poly_points = 64  # 对应数据预处理的采样点数
        
        # 1. 多边形坐标头: 输入 hidden_dim -> 输出 64 * 2 (x, y)
        self.poly_coord_embed = MLP(hidden_dim, hidden_dim, self.num_poly_points * 2, 3)
        
        # 2. 角点分类头: 输入 hidden_dim -> 输出 64 * 1 (logits)
        self.poly_corner_embed = MLP(hidden_dim, hidden_dim, self.num_poly_points, 3)
        
        # ===== [新增] Contour Evolution Module (BuildMapper-style) =====
        # 采样多尺度特征后 concat，通道数 = hidden_dim * num_feature_levels
        # 然后投影到 64D
        self.evo_point_feat_dim = 64
        
        # [SOTA优化] 输入维度增强: 
        # 2 (abs xy) + 64 (img feat) + 5 (geo feat: rel_prev, rel_next, angle) = 71
        self.evo_in_dim = 2 + self.evo_point_feat_dim + 5

        self.evo_feat_proj = nn.Sequential(
            nn.Linear(hidden_dim * num_feature_levels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.evo_point_feat_dim),
        )

        self.contour_evolver = ContourEvolutionModule(in_dim=self.evo_in_dim, mid_dim=128)

        # 演化迭代次数（对齐 BuildMapper 两次 evolution）
        self.num_evolve_iters = 2
        # ===============================================================

        # [新增] 初始化权重
        # 坐标头初始化为0，让初始预测收缩在参考点上
        nn.init.constant_(self.poly_coord_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.poly_coord_embed.layers[-1].bias.data, 0)
        
        # 角点头初始化为负值 (如 -4.6)，使初始 sigmoid 概率接近 0.01 (因为大部分点不是角点)
        nn.init.constant_(self.poly_corner_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.poly_corner_embed.layers[-1].bias.data, -4.6)

        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed

            # [新增] 多边形头也进行 Clone，每层独立
            self.poly_coord_embed = _get_clones(self.poly_coord_embed, num_pred)
            self.poly_corner_embed = _get_clones(self.poly_corner_embed, num_pred)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

            # [新增] 如果不开启 refine，则共享或使用 ModuleList (这里保持一致性使用 ModuleList)
            self.poly_coord_embed = nn.ModuleList([self.poly_coord_embed for _ in range(num_pred)])
            self.poly_corner_embed = nn.ModuleList([self.poly_corner_embed for _ in range(num_pred)])
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, samples: NestedTensor, mask_ratio: float = 0.0):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        # ================== [新增] MAE 分支逻辑 ==================
        mae_outputs = None
        if self.use_mae and mask_ratio > 0.0:
            # 1. 选取最后一层特征图进行 Mask (srcs[-1])
            # 根据 MRT 论文，只重建最高层语义特征
            src_to_mask = srcs[-1] # (B, C, H, W)
            B, C, H, W = src_to_mask.shape
            
            # 2. 生成随机 Mask
            # 展平为 (B, HW)
            num_tokens = H * W
            num_masked = int(mask_ratio * num_tokens)
            
            # 生成随机索引
            # rand_indices: (B, HW)
            rand_indices = torch.rand(B, num_tokens, device=src_to_mask.device).argsort(dim=1)
            mask_indices = rand_indices[:, :num_masked] # 要被 Mask 掉的索引
            keep_indices = rand_indices[:, num_masked:] # 保留的索引
            
            # 创建 Mask 矩阵 (B, H, W)
            # mask_token 应该是一个可学习的向量，或者直接填 0
            # 这里简单起见填 0
            src_masked = src_to_mask.flatten(2).clone() # (B, C, HW)
            
            # 将被 mask 的位置置为 0
            # 为了批量操作，我们可以利用 scatter 或者索引
            # 这里简单循环 B (效率稍低但逻辑清晰)
            binary_mask = torch.zeros(B, num_tokens, device=src_to_mask.device)
            for i in range(B):
                src_masked[i, :, mask_indices[i]] = 0
                binary_mask[i, mask_indices[i]] = 1 # 记录 mask 位置用于计算 Loss
            
            src_masked = src_masked.view(B, C, H, W)
            
            # 3. 将 Mask 后的特征送入 Transformer Encoder
            # 注意：这里我们需要仅仅跑 Encoder，而原始的 self.transformer() 是一次性跑完 Encoder+Decoder
            # 因此，我们可能需要修改 DeformableTransformer 类，或者在这里手动调用 encoder
            # 假设 transformer 有 .encoder 属性
            
            # 构造 encoder 输入
            # 仅对最后一层做 mask，其他层保持原样输入给 encoder 用于提供 context (MRT论文思路)
            srcs_for_enc = srcs[:-1] + [src_masked]
            
            # 调用 encoder (需要确保你的 transformer 暴露了 get_encoder_output 接口)
            # 或者标准 DeformableDETR transformer 调用方式:
            # memory = self.transformer.encoder(srcs, masks, pos_embeds)
            # 这里我们假设 self.transformer 内部拆分了 encoder/decoder，或者我们直接修改 transformer.forward
            
            # 暂时假设 self.transformer.encoder 可以独立调用
            # 如果没有，你需要去修改 deformable_transformer.py
            memory = self.transformer.forward_encoder(srcs_for_enc, masks, pos)
            
            # 4. 取出最后一层的 Encoder Output 对应的部分
            # Encoder 输出的是多尺度特征展平后的序列
            # 我们需要切片出最后一层特征
            # memory: (B, Sum(H_l * W_l), C)
            spatial_shapes = torch.as_tensor([s.shape[-2:] for s in srcs], device=src_to_mask.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
            
            last_lvl_start = level_start_index[-1]
            encoded_last_lvl = memory[:, last_lvl_start:, :] # (B, HW, C)
            
            # 5. 送入 MAE Decoder 进行重构
            rec_features = self.mae_decoder(encoded_last_lvl, align_h=H, align_w=W) # (HW, B, C)
            rec_features = rec_features.permute(1, 0, 2) # (B, HW, C)
            
            # 6. 计算 MAE Loss (只在此时计算并返回，不影响主流程)
            # 目标是原始特征图 src_to_mask (未经 Proj 或 经 Proj 均可，建议经 Proj)
            target_features = src_to_mask.flatten(2).permute(0, 2, 1) # (B, HW, C)
            
            loss_mae = F.mse_loss(rec_features, target_features, reduction='none')
            loss_mae = (loss_mae.mean(dim=-1) * binary_mask).sum() / (binary_mask.sum() + 1e-6)
            
            mae_outputs = {'loss_mae': loss_mae}
            
            # 如果只是为了跑 MAE 预热，可以直接 return mae_outputs
            # return mae_outputs 
        # =========================================================

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        # [新增] 列表存储多边形结果
        outputs_polys = []
        outputs_corners = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            # === [新增] 多边形预测逻辑 ===
            # 1. 预测坐标偏移 (Batch, Queries, 128)
            poly_offset = self.poly_coord_embed[lvl](hs[lvl])
            # Reshape 为 (Batch, Queries, 64, 2)
            poly_offset = poly_offset.view(poly_offset.shape[0], poly_offset.shape[1], self.num_poly_points, 2)
            
            # 2. 获取当前预测的 Box 中心 (在 inverse_sigmoid 空间)
            # tmp 包含了 [cx_logit, cy_logit, w_logit, h_logit]
            # ref_center = tmp[..., :2].unsqueeze(2) # (Batch, Queries, 1, 2)
            
            # === [修改] 使用 Box Template 初始化 ===
            # 2.1 获取预测框 (cx, cy, w, h)
            pred_boxes = tmp.sigmoid() # (B, Q, 4)
            cx = pred_boxes[..., 0:1].unsqueeze(2) # (B, Q, 1, 1)
            cy = pred_boxes[..., 1:2].unsqueeze(2)
            w  = pred_boxes[..., 2:3].unsqueeze(2)
            h  = pred_boxes[..., 3:4].unsqueeze(2)
            
            # 2.2 生成模板 (1, 1, 64, 2)
            if not hasattr(self, 'box_template'):
                self.box_template = get_box_template(self.num_poly_points, device=hs.device).view(1, 1, self.num_poly_points, 2)
            
            # 2.3 变换模板: template * (w, h) + (cx - w/2, cy - h/2)
            # 模板是 [0,1]x[0,1]，所以要先平移到 [-0.5, 0.5] 再缩放，或者直接缩放再平移左上角
            # 这里采用: 左上角 = (cx - w/2, cy - h/2)
            # poly = template * (w, h) + (cx - w/2, cy - h/2)
            
            # template[..., 0] 是 x, template[..., 1] 是 y
            # x_new = x_tpl * w + (cx - w/2)
            # y_new = y_tpl * h + (cy - h/2)
            
            box_init_poly = self.box_template * torch.cat([w, h], dim=-1) + torch.cat([cx - w/2, cy - h/2], dim=-1)
            
            # 2.4 转回 inverse_sigmoid 空间以便和 poly_offset 相加
            # 注意 clamp 防止数值不稳定
            box_init_poly_inv = inverse_sigmoid(box_init_poly.clamp(1e-4, 1 - 1e-4))
            
            # 3. 加上偏移量并 Sigmoid 归一化
            # 多边形坐标 = Sigmoid( Box初始多边形(inv) + 偏移量 )
            outputs_poly = (poly_offset + box_init_poly_inv).sigmoid()
            # ======================================
            
            outputs_polys.append(outputs_poly)

            # 4. 预测角点 (Batch, Queries, 64)
            outputs_corner = self.poly_corner_embed[lvl](hs[lvl])
            outputs_corners.append(outputs_corner)
            # ============================
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        # [新增] Stack
        outputs_poly = torch.stack(outputs_polys)
        outputs_corner = torch.stack(outputs_corners)

         # ===== [新增] Contour Evolution =====
        # 使用最后一层 decoder 的 init polygon 作为演化起点
        poly_init = outputs_poly[-1]          # (B,Q,N,2) in [0,1]
        corner_init_logits = outputs_corner[-1]  # (B,Q,N)

        polys_evolved = []
        vtx_logits_evolved = []

        poly_curr = poly_init
        for _it in range(self.num_evolve_iters):
            # 1) sample multiscale point features: (B,Q,N, sumC)
            feat_cat = sample_multiscale_point_features(srcs, poly_curr)

            # 2) project to 64D point feat
            B, Q, N, Csum = feat_cat.shape
            feat_64 = self.evo_feat_proj(feat_cat.view(B * Q * N, Csum)).view(B, Q, N, self.evo_point_feat_dim)

            # 3) build vertex feature = [x,y] + feat64 + geo_feat => 71D
            # [SOTA优化] 计算几何特征
            geo_feat = get_geometric_features(poly_curr) # (B, Q, N, 5)
            
            vtx_feat = torch.cat([poly_curr, feat_64, geo_feat], dim=-1)  # (B,Q,N,71)

            # 4) evolver expects (Bq, N, 71)
            delta, vtx_logits = self.contour_evolver(vtx_feat.view(B * Q, N, self.evo_in_dim))

            # 5) update polygon coordinates in logits space (more stable)
            poly_logits = inverse_sigmoid(poly_curr.clamp(1e-4, 1 - 1e-4))
            poly_logits = poly_logits + delta.view(B, Q, N, 2)

            poly_curr = poly_logits.sigmoid()
            polys_evolved.append(poly_curr)
            vtx_logits_evolved.append(vtx_logits.view(B, Q, N))
        # ===========================================================

        polys_evolved[0] = polys_evolved[0].clamp(0.0, 1.0)
        polys_evolved[1] = polys_evolved[1].clamp(0.0, 1.0)

        # 更新输出字典
        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1],

            # init
            'pred_polys_init': poly_init,
            'pred_corners_init': corner_init_logits,

            # evolved (two iterations)
            'pred_polys_evolve_0': polys_evolved[0],  # 第一次轮廓演化
            'pred_polys_evolve_1': polys_evolved[1],  # 第二次轮廓演化
            'pred_vtx_logits_evolve_0': vtx_logits_evolved[0],  # 第一次演化的顶点 logits
            'pred_vtx_logits_evolve_1': vtx_logits_evolved[1],  # 第二次演化的顶点 logits
        }

        if self.aux_loss:
            # 需要修改 _set_aux_loss 函数来支持新参数
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_poly, outputs_corner
            )

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        
        if mae_outputs is not None:
            out.update(mae_outputs)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_poly, outputs_corner):
        # 增加 poly 和 corner 的打包
        return [
            {
                'pred_logits': a,
                'pred_boxes': b,
                'pred_polys_init': c,
                'pred_corners_init': d
            }
            for a, b, c, d in zip(outputs_class[:-1], outputs_coord[:-1], outputs_poly[:-1], outputs_corner[:-1])
        ]

def densify_polygon_torch(poly: torch.Tensor, factor: int = 10) -> torch.Tensor:
    """
    poly: (M, N, 2) in [0,1], assumed ordered, NOT necessarily closed (we will close internally)
    return: (M, N*factor, 2) dense boundary points, each edge contributes `factor` points (endpoint excluded)
    """
    assert poly.dim() == 3 and poly.size(-1) == 2
    M, N, _ = poly.shape

    # close polygon: next point for last vertex is the first
    p0 = poly
    p1 = torch.roll(poly, shifts=-1, dims=1)  # (M,N,2)

    # t in [0,1) with `factor` steps, exclude endpoint to avoid duplicates across edges
    t = torch.linspace(0, 1, steps=factor, device=poly.device, dtype=poly.dtype, requires_grad=False)[:-1]
    # if factor=10, steps=10 -> remove last -> 9 points; we actually want 10 points/edge (paper says 10x).
    # So use steps=factor+1 then drop last:
    t = torch.linspace(0, 1, steps=factor + 1, device=poly.device, dtype=poly.dtype, requires_grad=False)[:-1]
    # t: (factor,)
    t = t.view(1, 1, factor, 1)  # (1,1,factor,1)

    p0e = p0.unsqueeze(2)  # (M,N,1,2)
    p1e = p1.unsqueeze(2)  # (M,N,1,2)
    dense = p0e + t * (p1e - p0e)  # (M,N,factor,2)
    dense = dense.reshape(M, N * factor, 2)
    return dense


def dml_losses(
    pred_poly: torch.Tensor,         # (M, N, 2)
    gt_poly: torch.Tensor,           # (M, N, 2)
    gt_corner_labels: torch.Tensor,  # (M, N) {0,1}
    densify_factor: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return:
      dml_bdy: pred -> dense boundary nearest (squared L2)
      dml_corner_pull: gt corners -> pred nearest (squared L2)
    """
    M, N, _ = pred_poly.shape
    if M == 0:
        return pred_poly.new_zeros(()), pred_poly.new_zeros(())

    gt_dense = densify_polygon_torch(gt_poly, factor=densify_factor)  # (M, N*factor, 2)

    # ---- (A) boundary pull: pred -> GT_dense nearest ----
    # cdist: (M, N, N*factor)
    dist = torch.cdist(pred_poly, gt_dense, p=2)
    min_dist, _ = dist.min(dim=-1)  # (M, N)
    dml_bdy = (min_dist ** 2).mean()  # mean over all vertices and instances

    # ---- (B) corner pull: GT corners -> pred nearest ----
    # variable #corners per instance => loop per instance (M is small, safe)
    corner_pull_sum = pred_poly.new_zeros(())
    corner_count = 0
    for m in range(M):
        mask = gt_corner_labels[m] > 0.5
        corners = gt_poly[m][mask]  # (K,2)
        if corners.numel() == 0:
            continue
        # (K,N)
        d = torch.cdist(corners.unsqueeze(0), pred_poly[m].unsqueeze(0), p=2)[0]
        min_d, _ = d.min(dim=1)  # (K,)
        corner_pull_sum = corner_pull_sum + (min_d ** 2).mean()
        corner_count += 1

    if corner_count > 0:
        dml_corner_pull = corner_pull_sum / corner_count
    else:
        dml_corner_pull = pred_poly.new_zeros(())

    return dml_bdy, dml_corner_pull


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

        self.epoch = 0

        # # ===== 课程 (先硬编码，后面可放到 args) =====
        # self.poly_warmup_epochs = 30          # 前30 epoch 主要靠 SmoothL1
        # self.poly_ramp_epochs = 10           # 接下来 10 epoch 把 DML 权重拉满
        # self.dml_densify_factor = 10         # GT densify 10x（BuildMapper）
        # # ===========================================
        # ===== 课程 (适配crowdai数据集) =====
        self.poly_warmup_epochs = 10          # 前10 epoch 主要靠 SmoothL1
        self.poly_ramp_epochs = 5           # 接下来 5 epoch 把 DML 权重拉满
        self.dml_densify_factor = 10         # GT densify 10x（BuildMapper）
        # ===========================================

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(
            src_logits, 
            target_classes_onehot, 
            alpha=self.focal_alpha, 
            gamma=2, 
            reduction='none'
        )
        loss_ce = loss_ce.mean(1).sum() / num_boxes * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    # 多边形 Loss 计算函数
    def loss_polys(self, outputs, targets, indices, num_boxes):
        # -------- hard guard: no gt in batch --------
        if num_boxes == 0:
            z = outputs['pred_logits'].new_zeros(())
            return {
                'loss_poly_smooth_init': z,
                'loss_poly_smooth_e0': z,
                'loss_dml_bdy': z,
                'loss_dml_corner_pull': z,
                'loss_poly_corner_init': z,
                'loss_vtx_corner_e0': z,
                'loss_vtx_corner_e1': z,
                'loss_poly_consistency': z # [新增]
            }

        idx = self._get_src_permutation_idx(indices)

        # ---- targets ----
        target_polys = torch.cat([t['poly_coords'][i] for t, (_, i) in zip(targets, indices)], dim=0)       # (M,64,2)
        target_corners = torch.cat([t['corner_labels'][i] for t, (_, i) in zip(targets, indices)], dim=0) # (M,64)

        # ---- predictions ----
        src_init = outputs['pred_polys_init'][idx]  # (M,64,2)
        src_e0 = outputs['pred_polys_evolve_0'][idx] if 'pred_polys_evolve_0' in outputs else None
        src_e1 = outputs['pred_polys_evolve_1'][idx] if 'pred_polys_evolve_1' in outputs else None

        src_corner_init = outputs['pred_corners_init'][idx] if 'pred_corners_init' in outputs else None
        src_vtx0 = outputs['pred_vtx_logits_evolve_0'][idx] if 'pred_vtx_logits_evolve_0' in outputs else None
        src_vtx1 = outputs['pred_vtx_logits_evolve_1'][idx] if 'pred_vtx_logits_evolve_1' in outputs else None

        # ---- curriculum weights分阶段课程权重 ----
        epoch = getattr(self, 'epoch', 0)
        
        # [SOTA优化] 统一课程学习策略 (Unified Curriculum Learning)
        # 阶段1 (0-30 epoch): 专注 SmoothL1 回归，让点先跑到轮廓附近。DML 和 Corner Loss 权重为 0。
        # 阶段2 (30-40 epoch): 线性过渡。SmoothL1 权重降低，DML 和 Corner Loss 权重逐渐拉满。
        # 阶段3 (40+ epoch): 精细化调整。SmoothL1 保持低权重，DML 和 Corner Loss 全力工作。
        
        if epoch <= self.poly_warmup_epochs:
            smooth_w = 1.0
            dml_w = 0.0
            corner_w = 0.0
        elif epoch <= self.poly_warmup_epochs + self.poly_ramp_epochs:
            t = (epoch - self.poly_warmup_epochs) / self.poly_ramp_epochs
            smooth_w = 1.0 - 0.7 * t   # 1.0 -> 0.3
            dml_w = t                  # 0.0 -> 1.0
            corner_w = t               # 0.0 -> 1.0
        else:
            smooth_w = 0.3
            dml_w = 1.0
            corner_w = 1.0

        losses = {}

        # ---- SmoothL1 (init) ----
        loss_init = F.smooth_l1_loss(src_init, target_polys, reduction='none', beta=0.1)
        losses['loss_poly_smooth_init'] = (loss_init.sum() / num_boxes) * smooth_w

        # ---- SmoothL1 (evolve_0) ----
        if src_e0 is not None:
            loss_e0 = F.smooth_l1_loss(src_e0, target_polys, reduction='none', beta=0.1)
            losses['loss_poly_smooth_e0'] = (loss_e0.sum() / num_boxes) * smooth_w
        else:
            losses['loss_poly_smooth_e0'] = src_init.new_zeros(())

        # ---- DML only on final evolve_1 ----
        if src_e1 is not None:
            dml_bdy, dml_corner_pull = dml_losses(
                pred_poly=src_e1,
                gt_poly=target_polys,
                gt_corner_labels=target_corners,
                densify_factor=getattr(self, 'dml_densify_factor', 10),
            )
            losses['loss_dml_bdy'] = dml_bdy * dml_w
            losses['loss_dml_corner_pull'] = dml_corner_pull * dml_w
        else:
            losses['loss_dml_bdy'] = src_init.new_zeros(())
            losses['loss_dml_corner_pull'] = src_init.new_zeros(())

        # ---- corner supervision (init head) ----
        # if src_corner_init is not None:
        #     loss_corner_init = sigmoid_focal_loss(
        #         src_corner_init.flatten(),
        #         target_corners.flatten(),
        #         alpha=0.25, gamma=2.0, reduction='none'
        #     )
        #     losses['loss_poly_corner_init'] = loss_corner_init.sum() / num_boxes
        # else:
        losses['loss_poly_corner_init'] = src_init.new_zeros(())

        # ---- vertex logits from evolution: supervise as CORNER logits (very important) ----
        # 这一步对抑制“长尖刺/爆炸”非常有效：让 evolution 的顶点特征知道哪些位置该“像角点”
        
        # [Fix] 动态匹配：因为预测点会移动，不能直接用 GT 的 corner_labels (它是对应 GT 顶点的)
        # 我们需要找到离 GT 角点最近的预测点，将其作为正样本
        def get_dynamic_corner_targets(pred_poly, gt_poly, gt_corner, dist_thresh=0.05):
            # pred_poly: (M, 64, 2)
            # gt_poly: (M, 64, 2)
            # gt_corner: (M, 64)
            M, N, _ = pred_poly.shape
            dynamic_targets = torch.zeros_like(gt_corner)
            
            for i in range(M):
                # 1. 提取当前 GT 的真实角点坐标
                mask = gt_corner[i] > 0.5
                if not mask.any():
                    continue
                real_corners = gt_poly[i][mask] # (K, 2)
                
                # 2. 计算预测点到真实角点的距离矩阵 (N, K)
                # pred_poly[i]: (N, 2)
                dists = torch.cdist(pred_poly[i].unsqueeze(0), real_corners.unsqueeze(0))[0]
                
                # 3. 对于每个真实角点，找到最近的预测点索引
                min_vals, min_idxs = dists.min(dim=0) # (K,)
                
                # [SOTA优化] 4. 增加距离阈值过滤
                # 只有当预测点距离真实角点足够近时，才标记为正样本
                valid_mask = min_vals < dist_thresh
                if valid_mask.any():
                    valid_idxs = min_idxs[valid_mask]
                    dynamic_targets[i, valid_idxs] = 1.0
            
            return dynamic_targets

        if src_vtx0 is not None:
            # 动态生成 target
            target_c0 = get_dynamic_corner_targets(src_e0.detach(), target_polys, target_corners)
            l0 = sigmoid_focal_loss(
                src_vtx0.flatten(), target_c0.flatten(),
                alpha=0.75, gamma=3.0, reduction='none'
            )
            losses['loss_vtx_corner_e0'] = (l0.sum() / num_boxes) * corner_w
        else:
            losses['loss_vtx_corner_e0'] = src_init.new_zeros(())

        if src_vtx1 is not None:
            # 动态生成 target
            target_c1 = get_dynamic_corner_targets(src_e1.detach(), target_polys, target_corners)
            l1 = sigmoid_focal_loss(
                src_vtx1.flatten(), target_c1.flatten(),
                alpha=0.75, gamma=3.0, reduction='none'
            )
            losses['loss_vtx_corner_e1'] = (l1.sum() / num_boxes) * corner_w

            # ========================= [在此处插入新增代码] =========================
            # 条件几何一致性 Loss (解决漏检 + 抑制直线上的冗余)
            
            # 1. 计算几何特征
            # src_e1: (B, N, 2) -> 最后一维第4项是 angle_cos
            geo_feat = get_geometric_features(src_e1) 
            pred_cos = geo_feat[..., 4] # 范围 [-1, 1]
            pred_prob = src_vtx1.sigmoid()

            # 2. 定义 "救援 (Rescue)" 掩码 -> 强行拉回漏检的直角
            # 条件: 物理上很尖 (cos < 0.25, 约75°~105°) AND 模型预测忽略了它 (prob < 0.4)
            rescue_mask = (pred_cos < 0.25) & (pred_prob < 0.4)

            # 3. 定义 "抑制 (Suppress)" 掩码 -> 强行按死直线上的冗余
            # 条件: 物理上很直 (cos > 0.95, 约10°以内) AND 模型错误预测为角点 (prob > 0.5)
            suppress_mask = (pred_cos > 0.95) & (pred_prob > 0.5)
            
            loss_consist = src_init.new_zeros(())
            
            # 救援 Loss (权重 10.0)
            if rescue_mask.any():
                loss_consist += F.mse_loss(pred_prob[rescue_mask], torch.ones_like(pred_prob[rescue_mask])) * 10.0
            
            # 抑制 Loss (权重 10.0)
            if suppress_mask.any():
                loss_consist += F.mse_loss(pred_prob[suppress_mask], torch.zeros_like(pred_prob[suppress_mask])) * 10.0

            losses['loss_poly_consistency'] = loss_consist * corner_w
            # ======================================================================

        else:
            losses['loss_vtx_corner_e1'] = src_init.new_zeros(())
            losses['loss_poly_consistency'] = src_init.new_zeros(())

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'polys': self.loss_polys  # [新增] 注册多边形 Loss(L1 + Corner )
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def circular_pad_1d(x: torch.Tensor, pad: int) -> torch.Tensor:
    """
    x: (B, C, N)
    pad: number of vertices to pad on both sides
    """
    if pad <= 0:
        return x
    left = x[:, :, -pad:]
    right = x[:, :, :pad]
    return torch.cat([left, x, right], dim=2)


class CircularConv1d(nn.Module):
    """
    1D conv with circular padding for polygon vertex sequence.
    """
    def __init__(self, in_ch, out_ch, k=3, d=1, groups=1, bias=True):
        super().__init__()
        self.k = k
        self.d = d
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=d, padding=0,
                              groups=groups, bias=bias)

    def forward(self, x):
        # x: (B,C,N)
        if self.k == 1:
            return self.conv(x)
        # effective padding for "same" length
        pad = (self.k - 1) // 2 * self.d
        x = circular_pad_1d(x, pad)
        return self.conv(x)


class Conv1dReluBN(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, d=1):
        super().__init__()
        self.conv = CircularConv1d(in_ch, out_ch, k=k, d=d)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ContourEvolutionModule(nn.Module):
    """
    BuildMapper Fig.6 style contour evolution module.
    Inputs:
      - vertex_feat: (B*Q, N, F)  where F = 66 by default (2 coords + 64 sampled feat)
    Outputs:
      - delta: (B*Q, N, 2) offset in logits-space (recommended) or xy-space
      - vtx_logits: (B*Q, N) vertex valid/invalid logits
    """
    def __init__(self, in_dim=66, mid_dim=128):
        super().__init__()

        # Up-dim: 66 -> 128 (k=1)
        self.up = Conv1dReluBN(in_dim, mid_dim, k=1)

        # Detail information (k=1) with residual adds
        self.detail1 = Conv1dReluBN(mid_dim, mid_dim, k=1)
        self.detail2 = Conv1dReluBN(mid_dim, mid_dim, k=1)
        self.detail3 = Conv1dReluBN(mid_dim, mid_dim, k=1)
        

        # Local information (k=3) residual
        self.local1 = Conv1dReluBN(mid_dim, mid_dim, k=3)
        self.local2 = Conv1dReluBN(mid_dim, mid_dim, k=3)

        # Global information (k=9, d=7) residual
        self.global_ = Conv1dReluBN(mid_dim, mid_dim, k=9, d=7)

        # Concat + fuse (k=1)
        self.fuse = Conv1dReluBN(mid_dim * 3, mid_dim, k=1)

        # Offset head: (k=1)->(k=1)->offset(2)
        self.off1 = Conv1dReluBN(mid_dim, mid_dim, k=1)
        self.off2 = Conv1dReluBN(mid_dim, mid_dim, k=1)
        self.off_out = nn.Conv1d(mid_dim, 2, kernel_size=1)

        # Vertex classification head: (k=1)->(k=1)->logit(1)
        self.cls1 = Conv1dReluBN(mid_dim, mid_dim, k=1)
        self.cls2 = Conv1dReluBN(mid_dim, mid_dim, k=1)
        self.cls_out = nn.Conv1d(mid_dim, 1, kernel_size=1)

    def forward(self, vertex_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # vertex_feat: (Bq, N, F) -> (Bq, F, N)
        x = vertex_feat.permute(0, 2, 1).contiguous()

        x0 = self.up(x)  # (Bq, 128, N)

        # detail block
        d1 = self.detail1(x0)
        d1 = d1 + x0
        d2 = self.detail2(d1)
        d2 = d2 + d1
        detail = d2

        # local block
        l1 = self.local1(detail)
        l1 = l1 + detail
        l2 = self.local2(l1)
        l2 = l2 + l1
        loc = l2

        # global block
        glo = self.global_(loc)
        glo = glo + loc

        # concat three info streams
        cat = torch.cat([detail, loc, glo], dim=1)  # (Bq, 384, N)
        enc = self.fuse(cat)  # (Bq, 128, N)

        # heads
        off = self.off2(self.off1(enc))
        delta = self.off_out(off).permute(0, 2, 1).contiguous()  # (Bq, N, 2)

        v = self.cls2(self.cls1(enc))
        vtx_logits = self.cls_out(v).squeeze(1)  # (Bq, N)

        return delta, vtx_logits


def sample_multiscale_point_features(
    srcs: List[torch.Tensor],  # each (B,C,H,W)
    polys: torch.Tensor,       # (B,Q,N,2) in [0,1]
) -> torch.Tensor:
    """
    Sample multi-scale features at polygon vertex locations.
    Return: (B,Q,N, C_total) where C_total = sum(C_l)
    """
    B, Q, N, _ = polys.shape
    # grid_sample expects grid in [-1,1]
    grid = polys * 2.0 - 1.0  # (B,Q,N,2)
    # grid_sample wants (B, H_out, W_out, 2). We'll use H_out=N, W_out=1
    grid = grid.view(B, Q * N, 1, 2)

    feats = []
    for s in srcs:
        # s: (B,C,H,W)
        f = F.grid_sample(s, grid, mode='bilinear', padding_mode='border', align_corners=False)
        # f: (B, C, Q*N, 1)
        f = f.squeeze(-1).permute(0, 2, 1).contiguous()  # (B, Q*N, C)
        feats.append(f)
    feat_cat = torch.cat(feats, dim=-1)  # (B, Q*N, sumC)
    feat_cat = feat_cat.view(B, Q, N, -1)
    return feat_cat


def build(args):
    num_classes = 1
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        use_mae=args.use_mae
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)

    # 权重
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef * 0.3
    
    # poly / evolution losses
    weight_dict['loss_poly_smooth_init'] = args.poly_coord_loss_coef
    weight_dict['loss_poly_smooth_e0'] = args.poly_coord_loss_coef
    # [修改] DML Loss 数值通常很小 (1e-4 量级)，需要给予更大的权重才能产生有效梯度
    # 这里给予 100 倍的额外增益，使其与 bbox loss (1e-2 量级) 处于同一数量级
    weight_dict['loss_dml_bdy'] = args.poly_coord_loss_coef * 100.0
    weight_dict['loss_dml_corner_pull'] = args.poly_coord_loss_coef * 100.0

    # corner losses (init + evolution vertex logits)
    weight_dict['loss_poly_corner_init'] = args.poly_corner_loss_coef
    weight_dict['loss_vtx_corner_e0'] = args.poly_corner_loss_coef
    weight_dict['loss_vtx_corner_e1'] = args.poly_corner_loss_coef
    weight_dict['loss_poly_consistency'] = args.poly_corner_loss_coef # [新增] 注册 consistency loss 权重

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        # 注意：Enc outputs 通常只做检测，不预测多边形，所以不用复制 poly 权重到 enc
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items() if 'poly' not in k})
        weight_dict.update(aux_weight_dict)

    # [新增] 添加 'polys' 到 loss 列表
    losses = ['labels', 'boxes', 'cardinality', 'polys']
    if args.masks:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors


class AuxiliaryMAEDecoder(nn.Module):
    """
    轻量级 MAE Decoder，用于从被 Mask 的 Encoder 特征中重构原始特征。
    参考论文 MRT 附录 Table 2 的参数设置。
    """
    def __init__(self, d_model=256, nhead=8, num_layers=2, feature_size=(21, 42)):
        super().__init__()
        # 这里的 feature_size 是最后一层特征图的大小
        # 假设输入图像 666x1333 -> ResNet下采样32倍 -> 约 21x42 (882个Queries)
        self.feature_size = feature_size
        self.num_queries = feature_size[0] * feature_size[1]
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=1024, dropout=0.1)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # 固定位置的 Queries，代表特征图上的每个 Grid
        self.query_embed = nn.Embedding(self.num_queries, d_model)
        
        # 输出投影层，还原到 d_model 维度
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x_masked, align_h=None, align_w=None):
        # x_masked: (B, SeqLen_Enc, C)
        bs = x_masked.shape[0]
        
        # 1. 准备 tgt (Queries)
        # 原始定义的固定尺寸 (10, 10)
        h_ref, w_ref = self.feature_size
        
        # 将 Query Embed (100, 256) 还原为 2D 空间 (1, 256, 10, 10)
        # 注意 query_embed 是 (Num_Queries, C)，我们需要 permute
        query_embed = self.query_embed.weight.permute(1, 0).view(1, -1, h_ref, w_ref)
        
        # === [关键修复] 动态插值 ===
        # 如果当前特征图尺寸 (align_h, align_w) 与预设不一致，则强制插值对齐
        if align_h is not None and (align_h != h_ref or align_w != w_ref):
            query_embed = F.interpolate(
                query_embed, 
                size=(align_h, align_w), 
                mode='bilinear', 
                align_corners=False
            )
        
        # 展平回序列: (1, 256, H_new, W_new) -> (1, 256, SeqLen_New) -> (SeqLen_New, 1, 256)
        tgt = query_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
        
        # 2. 准备 memory (x_masked)
        if x_masked.dim() == 4:
            x_masked = x_masked.flatten(2).permute(2, 0, 1)
        elif x_masked.dim() == 3 and x_masked.shape[1] != bs: 
             x_masked = x_masked.permute(1, 0, 2)

        # 3. Transformer 解码
        output = self.decoder(tgt, x_masked)
        
        # 4. 投影回特征空间
        return self.output_proj(output)