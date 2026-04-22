import torch
import torch_scatter
import torch.nn as nn
import numpy as np
from model.voxel_fea_generator import voxel_3d_generator, voxelization
from util.model_util import ResConvBlock, ConvBlock, DownBlock, RegBlock, Pool, Conv, ResConv
from model.sphereformer import Semantic
from model.loss import CriterionCoordinate
import torch.nn.functional as F
# from OpenPCDet.pcdet.models.backbones_2d.base_bev_backbone import BEVNet
from OpenPCDet.pcdet.models.backbones_2d.base_bev_backbone import BaseBEVResBackbone
from OpenPCDet.pcdet.models.dense_heads import CenterHead,AnchorHeadSingle
import pdb
import copy 

import spconv.pytorch as spconv

error_points={}

class attention_func(nn.Module):
    def __init__(self, l_ch, s_ch, m_ch, cat_ch, scale_list, strides, spatial_shape):
        super(attention_func, self).__init__()
        """
        l_ch: large feats 最上层特征维度
        s_ch: small feats 中间层特征维度
        m_ch: master feats 最下层特征维度
        """
        self.scale_list = scale_list
        self.strides = strides
        self.spatial_shape = spatial_shape
        self.l_pool = Pool(self.scale_list[2], self.scale_list[0], np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())
        self.s_pool = Pool(self.scale_list[2], self.scale_list[1], np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())

        self.l_conv = Conv(l_ch, cat_ch, kernel_size=1, indice_key='l_conv')
        self.s_conv = Conv(s_ch, cat_ch, kernel_size=1, indice_key='s_conv')

        self.l_squeeze = ResConv(cat_ch, 1, kernel_size=1, indice_key='l_squeeze')
        self.s_squeeze = ResConv(cat_ch, 1, kernel_size=1, indice_key='s_squeeze')
        self.m_squeeze = ResConv(m_ch, 1, kernel_size=1, indice_key='m_squeeze')

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(inplace=True)
        self.file_counter = 0


    def forward(self, s_feats, l_feats, m_feats, data_dict):
        l_feats = self.l_pool(l_feats, data_dict)
        l_feats = self.l_conv(l_feats)
        s_feats = self.s_pool(s_feats, data_dict)
        s_feats = self.s_conv(s_feats)
        _s_feats = self.s_squeeze(s_feats)
        _l_feats = self.l_squeeze(l_feats)
        out_feats = self.m_squeeze(m_feats)
        att_map_sum = self.sigmoid(out_feats.features + _l_feats.features)
        att_map_mul = self.sigmoid(out_feats.features + _s_feats.features)
        out_feats = out_feats.replace_feature(torch.cat((m_feats.features, l_feats.features * att_map_sum,
                                                         s_feats.features * att_map_mul), 1))
        data_dict['sparse_tensor'] = out_feats
        return data_dict

class HeightCompression(nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self, data_dict):
        encoded_spconv_tensor = data_dict['sparse_tensor']           #SparseConvTensor[shape=torch.Size([887, 128])]  .spatial_shape[5, 75, 75]
        print(data_dict['sparse_tensor'].spatial_shape)
        indices = encoded_spconv_tensor.indices
        
        if not (indices[:,1].max() < 5 and indices[:,2].max()<75 and  indices[:,3].max()<75):
            global error_points
            np.save("/home/data/code/anchor_SG/results/error_points.npy",error_points)

        encoded_spconv_tensor.spatial_shape = [5, 75, 75] # [4, 60, 60]
        spatial_features = encoded_spconv_tensor.dense()             #torch.Size([1, 128, 5, 75, 75])         
        N, C, D, H, W = spatial_features.shape                       #N=1是batchsize，C=128是通道，D=5是深度z，H是高度，W是宽度         #kitti的是torch.Size([10, 128, 2, 200, 176])
        spatial_features = spatial_features.view(N, C * D, H, W)     #torch.Size([1, 640, 75, 75])       将2、3维合并成一个新的维度。
        print(spatial_features.shape) # torch.Size([1, 256, 60, 60])
        data_dict['spatial_features'] = spatial_features
        data_dict['spatial_features_stride'] = 8
        return data_dict

class AlignedContrastiveLoss(nn.Module):
    def __init__(self, topk=10, threshold=0.5, margin=0.5):
        super().__init__()
        self.topk = topk          # 选择前 k 个高亮点
        self.threshold = threshold  # 特征响应阈值
        self.margin = margin      # 相似度 margin（希望相似度 < margin）
    
    def find_peaks(self, feature_map):
        C, H, W = feature_map.shape
        with torch.no_grad():  # 不需要梯度计算
            intensity = torch.norm(feature_map, dim=0, p=2)  # (H, W)
            
            # 使用 max_pool2d 快速查找局部最大值
            pooled = F.max_pool2d(intensity.unsqueeze(0), 3, stride=1, padding=1)
            peak_mask = (intensity == pooled.squeeze(0)) & (intensity > self.threshold)
            
            # 直接获取坐标
            y, x = torch.where(peak_mask)
            indices = torch.argsort(intensity[y, x], descending=True)
            
            # 取 topk
            topk = min(self.topk, len(y))
            return list(zip(y[indices[:topk]].cpu().numpy(), 
                            x[indices[:topk]].cpu().numpy()))
    
    # def find_peaks(self, feature_map):
    #     """在 BEV 特征图上寻找高响应点（峰值）"""
    #     C, H, W = feature_map.shape
    #     feature_map = F.normalize(feature_map, dim=0, p=2)
        
    #     # 计算每个空间位置的特征强度（L2 范数）
    #     intensity = torch.norm(feature_map, dim=0)  # (H, W)
        
    #     # 找到 topk 个最高响应点
    #     flat_intensity = intensity.flatten()
    #     topk_values, topk_indices = torch.topk(flat_intensity, k=self.topk)
        
    #     # 转换为 (y, x) 坐标
    #     peaks = []
    #     for idx in topk_indices:
    #         if flat_intensity[idx] > self.threshold:  # 仅保留高于阈值的点
    #             y = idx // W
    #             x = idx % W
    #             peaks.append((y, x))
        
    #     return peaks

    def extract_feature_at_peaks(self, feature_map, peaks):
        """在峰值位置提取特征向量"""
        features = []
        for (y, x) in peaks:
            features.append(feature_map[:, y, x])  # (C,)
        if features:
            return torch.stack(features, dim=0)  # (N_peaks, C)
        else:
            return torch.zeros((0, feature_map.shape[0]), device=feature_map.device)

    def forward(self, data_dict):
        loc_features, det_features = data_dict['contrast_data']
        gt_boxes = data_dict["gt_boxes"]
        contrastive_loss = 0

        for ba in range(loc_features.shape[0]):  # batch_size
            # 提取检测特征（基于 det_features 的高亮点）
            peaks = self.find_peaks(det_features[ba])
            det_features_batch = self.extract_feature_at_peaks(det_features[ba], peaks)  # (N_peaks, C)

            if det_features_batch.shape[0] == 0:
                continue  # 如果没有高亮点，跳过

            # 提取定位特征（基于 loc_features 的高亮点）
            peaks = self.find_peaks(loc_features[ba])
            loc_features_batch = self.extract_feature_at_peaks(loc_features[ba], peaks)  # (N_peaks, C)

            if loc_features_batch.shape[0] == 0:
                continue  # 如果没有高亮点，跳过

            # 计算对比损失（鼓励检测和定位特征拉远）
            similarity = F.cosine_similarity(
                det_features_batch.unsqueeze(1),   # (1, N_peaks, C)
                loc_features_batch.unsqueeze(0),   # (1, N_peaks, C)
                dim=-1
            )
            # 希望相似度 < margin（鼓励不相关）
            loss = F.relu(similarity - self.margin).mean()
            contrastive_loss += loss

        return contrastive_loss / loc_features.shape[0]  # 平均 batch loss

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.cos_sim = nn.CosineSimilarity(dim=-1)
    
    def forward(self, static_features, dynamic_features):
        # 将静态和动态特征合并计算相似度矩阵
        features = torch.cat([static_features, dynamic_features], dim=0)
        sim_matrix = self.cos_sim(features.unsqueeze(1), features.unsqueeze(0)) / self.temperature
        
        # 构建标签：静态和动态特征互为positive
        labels = torch.zeros((len(features),), dtype=torch.long).to(features.device)
        num_static = static_features.size(0)
        num_dynamic = dynamic_features.size(0)
        
        # 静态特征的positive是动态特征，反之亦然
        for i in range(num_static):
            labels[i] = num_static + torch.randint(0, num_dynamic, (1,))
        for j in range(num_dynamic):
            labels[num_static + j] = torch.randint(0, num_static, (1,))
        
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

class DetContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, margin=0.2):
        super().__init__()
        self.point_cloud_range = [-59.9, -59.9, -2, 59.9, 59.9, 5.9]
        self.temperature = temperature  # 温度系数，控制相似度分布
        self.margin = margin  # 类别间区分的边界值
        # 类别映射（假设3类：Car=0, Pedestrian=1, Cyclist=2）
        self.class_names = ['Car', 'Pedestrian', 'Cyclist']
        self.num_classes = len(self.class_names)

    def extract_feature_at_box(self, bev_feature, box):
        """从BEV特征图中提取GT框中心的特征"""
        bounding = self.point_cloud_range
        begin_w = bounding[3] - bounding[0]  # x轴范围
        begin_h = bounding[4] - bounding[1]  # y轴范围
        x, y = box[0], box[1]  # 框中心坐标
        C, H, W = bev_feature.shape  # BEV特征图形状

        # 将3D坐标映射到BEV特征图坐标
        center_x = ((x - bounding[0]) / begin_w) * W
        center_y = ((y - bounding[1]) / begin_h) * H
        # 坐标裁剪（避免越界）
        center_x = torch.clamp(center_x, 0, W-1).long()
        center_y = torch.clamp(center_y, 0, H-1).long()

        return bev_feature[:, center_y, center_x]  # (C,)

    def forward(self, data_dict):
        """
        动态/静态跨类别对比损失计算：
        - 动态物体特征：正样本=同类静态，负样本=其他类静态
        - 静态物体特征：正样本=同类动态，负样本=其他类动态
        """
        # 输入特征：BEV检测特征 (B, C, H, W)
        spatial_features_2d = data_dict['spatial_features_2d']
        # GT框：(B, N, 8)，格式为[x, y, z, h, w, l, yaw, cls]，cls=0/1/2对应三类
        gt_boxes = data_dict["gt_boxes"]
        B = spatial_features_2d.shape[0]

        # 按类别和运动状态分组存储特征
        # 结构：{cls: {'static': [], 'dynamic': []}}
        class_feats = {c: {'static': [], 'dynamic': []} for c in range(self.num_classes)}

        for ba in range(B):
            bev_feat = spatial_features_2d[ba]  # (C, H, W)
            boxes = gt_boxes[ba]  # (N, 8)

            for box in boxes:
                cls = int(box[-1])  # 类别标签（0/1/2）
                if cls not in class_feats:
                    continue  # 跳过未知类别

                # 运动状态：假设box第7维为运动状态（0=静态，1=动态）
                # 若原始数据无运动状态标签，可根据速度等信息判断
                is_dynamic = bool(box[7]) if len(box) > 7 else False
                state = 'dynamic' if is_dynamic else 'static'

                # 提取框中心特征并加入对应分组
                feat = self.extract_feature_at_box(bev_feat, box)
                class_feats[cls][state].append(feat.unsqueeze(0))  # 增加batch维度

        # 转换为张量（过滤空列表）
        for c in class_feats:
            for state in ['static', 'dynamic']:
                if class_feats[c][state]:
                    class_feats[c][state] = torch.cat(class_feats[c][state], dim=0)  # (N, C)
                else:
                    class_feats[c][state] = torch.tensor([], device=spatial_features_2d.device)

        total_loss = 0.0
        count = 0  # 统计有效损失项数量

        # 1. 处理动态物体特征：正样本=同类静态，负样本=其他类静态
        for cls in range(self.num_classes):
            dynamic_feats = class_feats[cls]['dynamic']  # (N_d, C)
            same_static_feats = class_feats[cls]['static']  # 同类静态（正样本）(N_s, C)

            if len(dynamic_feats) == 0 or len(same_static_feats) == 0:
                continue  # 无足够样本，跳过

            # 收集其他类静态特征（负样本）
            other_static_feats = []
            for other_cls in range(self.num_classes):
                if other_cls != cls and len(class_feats[other_cls]['static']) > 0:
                    other_static_feats.append(class_feats[other_cls]['static'])
            if not other_static_feats:
                continue  # 无其他类静态样本，跳过
            other_static_feats = torch.cat(other_static_feats, dim=0)  # (N_o, C)

            # 特征归一化
            dynamic_feats = F.normalize(dynamic_feats, dim=-1)
            same_static_feats = F.normalize(same_static_feats, dim=-1)
            other_static_feats = F.normalize(other_static_feats, dim=-1)

            # 计算相似度：动态特征与同类静态（正）、其他类静态（负）
            sim_pos = torch.matmul(dynamic_feats, same_static_feats.T) / self.temperature  # (N_d, N_s)
            sim_neg = torch.matmul(dynamic_feats, other_static_feats.T) / self.temperature  # (N_d, N_o)

            # 每个动态样本的最佳正相似度和最差负相似度（硬样本挖掘）
            max_sim_pos, _ = sim_pos.max(dim=1)  # (N_d,)
            max_sim_neg, _ = sim_neg.max(dim=1)  # (N_d,)

            # 动态-静态对比损失：推远负样本，拉近正样本
            loss = F.relu(self.margin + max_sim_neg - max_sim_pos).mean()
            total_loss += loss
            count += 1

        # 2. 处理静态物体特征：正样本=同类动态，负样本=其他类动态
        for cls in range(self.num_classes):
            static_feats = class_feats[cls]['static']  # (N_s, C)
            same_dynamic_feats = class_feats[cls]['dynamic']  # 同类动态（正样本）(N_d, C)

            if len(static_feats) == 0 or len(same_dynamic_feats) == 0:
                continue  # 无足够样本，跳过

            # 收集其他类动态特征（负样本）
            other_dynamic_feats = []
            for other_cls in range(self.num_classes):
                if other_cls != cls and len(class_feats[other_cls]['dynamic']) > 0:
                    other_dynamic_feats.append(class_feats[other_cls]['dynamic'])
            if not other_dynamic_feats:
                continue  # 无其他类动态样本，跳过
            other_dynamic_feats = torch.cat(other_dynamic_feats, dim=0)  # (N_o, C)

            # 特征归一化
            static_feats = F.normalize(static_feats, dim=-1)
            same_dynamic_feats = F.normalize(same_dynamic_feats, dim=-1)
            other_dynamic_feats = F.normalize(other_dynamic_feats, dim=-1)

            # 计算相似度：静态特征与同类动态（正）、其他类动态（负）
            sim_pos = torch.matmul(static_feats, same_dynamic_feats.T) / self.temperature  # (N_s, N_d)
            sim_neg = torch.matmul(static_feats, other_dynamic_feats.T) / self.temperature  # (N_s, N_o)

            # 每个静态样本的最佳正相似度和最差负相似度
            max_sim_pos, _ = sim_pos.max(dim=1)  # (N_s,)
            max_sim_neg, _ = sim_neg.max(dim=1)  # (N_s,)

            # 静态-动态对比损失
            loss = F.relu(self.margin + max_sim_neg - max_sim_pos).mean()
            total_loss += loss
            count += 1

        # 平均有效损失项
        if count == 0:
            return torch.tensor(0.0, device=spatial_features_2d.device)
        return total_loss / count

class LocContrastiveLoss(nn.Module):
    def __init__(self, topk_true_geo=20, topk_static_obj=15, num_amb=30, 
                 temperature=0.1, margin=0.3):
        super().__init__()
        self.point_cloud_range = [-59.9, -59.9, -2, 59.9, 59.9, 5.9]
        
        # 采样参数
        self.topk_true_geo = topk_true_geo  # 真实地理特征的topk高响应点
        self.topk_static_obj = topk_static_obj  # 静态物体特征的topk采样数
        self.num_amb = num_amb  # 模糊特征的随机采样数
        
        # 对比参数
        self.temperature = temperature  # 温度系数
        self.margin = margin  # 正负样本距离边界
        
    def extract_true_geo_features(self, loc_features):
        """提取真实地理特征（正样本）：BEV图中高响应的静态地理结构点"""
        B, C, H, W = loc_features.shape
        true_geo_feats = []
        
        for ba in range(B):
            # 计算特征响应强度（L2范数）
            intensity = torch.norm(loc_features[ba], dim=0)  # (H, W)
            # 局部最大值筛选（过滤孤立噪声点）
            pooled = F.max_pool2d(intensity.unsqueeze(0), 3, stride=1, padding=1)
            peak_mask = (intensity == pooled.squeeze(0))
            
            # 提取topk高响应点特征
            y, x = torch.where(peak_mask)
            if len(y) == 0:
                # 无峰值时取全局topk
                flat_intensity = intensity.flatten()
                topk_indices = torch.topk(flat_intensity, k=min(self.topk_true_geo, len(flat_intensity)))[1]
                y = topk_indices // W
                x = topk_indices % W
            
            # 取topk特征
            topk = min(self.topk_true_geo, len(y))
            indices = torch.argsort(intensity[y, x], descending=True)[:topk]
            for idx in indices:
                feat = loc_features[ba, :, y[idx], x[idx]]
                true_geo_feats.append(feat.unsqueeze(0))
        
        return torch.cat(true_geo_feats, dim=0) if true_geo_feats else torch.tensor([], device=loc_features.device)
    
    def extract_static_obj_features(self, loc_features, gt_boxes):
        """提取静态物体特征（负样本）：GT标注的静态物体中心特征"""
        B, C, H, W = loc_features.shape
        static_obj_feats = []
        bounding = self.point_cloud_range
        begin_w = bounding[3] - bounding[0]
        begin_h = bounding[4] - bounding[1]
        
        for ba in range(B):
            # 筛选静态物体GT框（标签为0）
            static_boxes = gt_boxes[ba][gt_boxes[ba][:, -1] == 0]
            if len(static_boxes) == 0:
                continue
            
            # 采样topk静态物体（避免数量过多）
            sample_num = min(self.topk_static_obj, len(static_boxes))
            sample_indices = torch.randperm(len(static_boxes))[:sample_num]
            
            # 提取物体中心特征
            for idx in sample_indices:
                box = static_boxes[idx]
                x, y = box[0], box[1]
                # 转换为BEV特征图坐标
                center_x = ((x - bounding[0]) / begin_w) * W
                center_y = ((y - bounding[1]) / begin_h) * H
                # 坐标裁剪（避免越界）
                center_x = torch.clamp(center_x, 0, W-1).long()
                center_y = torch.clamp(center_y, 0, H-1).long()
                
                feat = loc_features[ba, :, center_y, center_x]
                static_obj_feats.append(feat.unsqueeze(0))
        
        return torch.cat(static_obj_feats, dim=0) if static_obj_feats else torch.tensor([], device=loc_features.device)
    
    def sample_ambiguous_features(self, loc_features):
        """采样模糊特征（查询样本）：随机选取BEV图中的特征点"""
        B, C, H, W = loc_features.shape
        amb_feats = []
        
        for ba in range(B):
            # 随机采样坐标
            sample_indices = torch.randperm(H * W)[:self.num_amb]
            y = sample_indices // W
            x = sample_indices % W
            
            # 提取特征
            for idx in range(self.num_amb):
                feat = loc_features[ba, :, y[idx], x[idx]]
                amb_feats.append(feat.unsqueeze(0))
        
        return torch.cat(amb_feats, dim=0) if amb_feats else torch.tensor([], device=loc_features.device)
    
    def forward(self, data_dict):
        """
        三元对比损失计算：
        1. 拉近查询样本与正样本距离
        2. 推远查询样本与负样本距离
        """
        loc_features = data_dict['aligned_loc_feature']  # 定位特征 (B, C, H, W)
        gt_boxes = data_dict["gt_boxes"]  # GT框 (B, N, 8)，最后一维为标签（0=静态，1=动态）
        
        # 1. 提取三类特征
        true_geo_feats = self.extract_true_geo_features(loc_features)  # (N_t, C)
        static_obj_feats = self.extract_static_obj_features(loc_features, gt_boxes)  # (N_s, C)
        amb_feats = self.sample_ambiguous_features(loc_features)  # (N_a, C)
        
        # 无有效特征时返回0损失
        if len(true_geo_feats) == 0 or len(static_obj_feats) == 0 or len(amb_feats) == 0:
            return torch.tensor(0.0, device=loc_features.device)
        
        # 2. 归一化特征（提升对比稳定性）
        true_geo_feats = F.normalize(true_geo_feats, dim=-1)
        static_obj_feats = F.normalize(static_obj_feats, dim=-1)
        amb_feats = F.normalize(amb_feats, dim=-1)
        
        # 3. 计算相似度矩阵
        # 查询样本与正样本相似度 (N_a, N_t)
        sim_pos = torch.matmul(amb_feats, true_geo_feats.T) / self.temperature
        # 查询样本与负样本相似度 (N_a, N_s)
        sim_neg = torch.matmul(amb_feats, static_obj_feats.T) / self.temperature
        
        # 4. 三元组损失计算（硬负样本挖掘）
        # 每个查询样本的最佳正样本相似度
        max_sim_pos, _ = sim_pos.max(dim=1, keepdim=True)
        # 每个查询样本的最差负样本相似度（硬负样本）
        max_sim_neg, _ = sim_neg.max(dim=1, keepdim=True)
        
        # 损失：max(0, margin + max_sim_neg - max_sim_pos)
        triplet_loss = F.relu(self.margin + max_sim_neg - max_sim_pos).mean()
        
        print(f"loc_contrastive_loss: {triplet_loss.item():.4f}")
        return triplet_loss

class LiSA(nn.Module):
    def __init__(self, config):
        super(LiSA, self).__init__()
        "Initialization# 初始化网络层"
        self.input_dim = config.input_c
        self.output_dim = config.output_c
        self.att_dim = config.att_dim
        self.conv_dim = config.layers
        self.scale_list = config.scale_list
        self.num_scale = len(self.scale_list)
        min_volume_space = config.min_volume_space
        max_volume_space = config.max_volume_space
        self.coors_range_xyz = [[min_volume_space[0], max_volume_space[0]],
                                [min_volume_space[1], max_volume_space[1]],
                                [min_volume_space[2], max_volume_space[2]]]
        self.spatial_shape = np.array(config.spatial_shape)
        self.model_cfg=config
        self.strides = [int(scale) for scale in self.scale_list]   
        self.file_counter = 0

        self.voxelizer = voxelization(
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape,
            scale_list=self.scale_list
        )

        self.voxel_3d_generator = voxel_3d_generator(
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape
        )

        self.conv1a = ConvBlock(in_channels=self.input_dim,
                                out_channels=self.conv_dim[0],
                                kernel_size=3,
                                indice_key='conv1a',
                                spatial_shape=np.int32(self.spatial_shape))

        self.conv1b = DownBlock(in_channels=self.conv_dim[0],
                                out_channels=self.conv_dim[1],
                                kernel_size=3,
                                indice_key='conv1b',
                                scale=self.scale_list[0],              #2
                                last_scale=1,
                                spatial_shape=np.int32(self.spatial_shape // self.strides[0])[::-1].tolist())

        self.conv2a = DownBlock(in_channels=self.conv_dim[1],
                                out_channels=self.conv_dim[2],
                                kernel_size=3,
                                indice_key='conv2a',
                                scale=self.scale_list[1],              #4
                                last_scale=self.scale_list[0],
                                spatial_shape=np.int32(self.spatial_shape // self.strides[1])[::-1].tolist())

        self.conv3a = ResConvBlock(in_channels=self.conv_dim[2],
                                   out_channels=self.conv_dim[3],
                                   kernel_size=3,
                                   indice_key='conv3a',
                                   spatial_shape=np.int32(self.spatial_shape // self.strides[1])[::-1].tolist())

        self.conv3b = DownBlock(in_channels=self.conv_dim[3],
                                out_channels=self.conv_dim[4],
                                kernel_size=3,
                                indice_key='conv3b',
                                scale=self.scale_list[2],
                                   last_scale=self.scale_list[1],          #8
                                spatial_shape=np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())

        self.conv4a = ResConvBlock(in_channels=self.conv_dim[4],
                                   out_channels=self.conv_dim[5],
                                   kernel_size=3,
                                   indice_key='conv4a',
                                   spatial_shape=np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())

        self.conv4b = ResConvBlock(in_channels=self.conv_dim[5],
                                   out_channels=self.conv_dim[6],
                                   kernel_size=3,
                                   indice_key='conv4b',
                                   spatial_shape=np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())

        self.conv4c = ResConvBlock(in_channels=self.conv_dim[6],
                                   out_channels=self.conv_dim[7],
                                   kernel_size=3,
                                   indice_key='conv4c',
                                   spatial_shape=np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())

        self.conv5a = ResConvBlock(in_channels=self.conv_dim[7] + 2 * self.att_dim,
                                   out_channels=self.conv_dim[8],
                                   kernel_size=1,
                                   indice_key='conv5a',
                                   spatial_shape=np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())

        self.conv5b = ResConvBlock(in_channels=self.conv_dim[8],
                                   out_channels=self.conv_dim[9],
                                   kernel_size=1,
                                   indice_key='conv5b',
                                   spatial_shape=np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())

        # self.convout0 = RegBlock(in_channels=self.conv_dim[9],
        #                         out_channels=self.output_dim,
        #                         indice_key='convout0',
        #                         spatial_shape=np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())

        self.convout = RegBlock(in_channels=self.conv_dim[9], #+ 64, # 3500+64
                                out_channels=self.output_dim,
                                indice_key='convout',
                                spatial_shape=np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())

        # self.convout_seg = RegBlock(in_channels=self.conv_dim[9],
        #                         out_channels=32,
        #                         indice_key='convout_seg',
        #                         spatial_shape=np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())
        # convbev
        self.convbev = ResConvBlock(in_channels=self.conv_dim[9],
                                   out_channels=64,
                                   kernel_size=1,
                                   indice_key='convbev',
                                   spatial_shape=np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())

        self.loc_convbev = ResConvBlock(in_channels=self.conv_dim[9],
                                   out_channels=64,
                                   kernel_size=1,
                                   indice_key='loc_convbev',
                                   spatial_shape=np.int32(self.spatial_shape // self.strides[2])[::-1].tolist())

        self.msf = attention_func(128, 256, 512, 128, self.scale_list, self.strides, self.spatial_shape)
        
        # detection
        self.maptobev = HeightCompression()
        self.backbone2d = BaseBEVResBackbone(self.model_cfg.BACKBONE_2D, input_channels=320)

        # 新增backbone用于定位特征对齐（需确保配置相同）
        self.loc_bev_backbone = BaseBEVResBackbone(self.model_cfg.BACKBONE_2D, input_channels=320)
        
        #class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer','barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        # class_names = ['car', 'bicycle', 'pedestrian'],
        #class_names= [ 'Car', 'Pedestrian', 'Cyclist' ]
        self.AnchorHeadSingle = AnchorHeadSingle(
            model_cfg=self.model_cfg.DENSE_HEAD,
            input_channels=128,
            num_class=3,
            class_names = ['Car', 'Pedestrian', 'Cyclist'],
            grid_size=np.array([600, 600, 40]),
            point_cloud_range=np.array([ -60 , -60 , -2 , 60 , 60 , 6 ]),
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),
            voxel_size=[0.2, 0.2, 0.2]
        )
        # self.AnchorHeadSingle = AnchorHeadSingle(
        #     model_cfg=self.model_cfg.DENSE_HEAD,
        #     input_channels=128,
        #     num_class=3,
        #     class_names = ['Car', 'Pedestrian', 'Cyclist'],
        #     # grid_size=np.array([480, 480, 32]),#np.array([1200, 1200, 80]), # np.array([600, 600, 40]),
        #     grid_size=np.array([1200, 1200, 80]),
        #     point_cloud_range=np.array([-60 , -60 , -2 , 60 , 60 , 6]),
        #     predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),
        #     # voxel_size=[0.25, 0.25, 0.25]#[0.1, 0.1, 0.1] # [0.2, 0.2, 0.2]
        #     voxel_size=[0.1, 0.1, 0.1]
        # )

        self.contrast_loss = AlignedContrastiveLoss()

        # 增加无参数的特征对齐层
        self.feature_align = nn.AdaptiveAvgPool2d((75, 75))  # 对齐到检测特征尺寸
        
    def forward(self, data_dict):
        global error_points
        error_points=copy.deepcopy(data_dict)
        assert data_dict["gt_boxes"].shape[0] > 0, "gt_boxes is empty"
        # data_dict=np.load("/home/data/code/anchor_SG/results/error_points.npy",allow_pickle=True)
        # data_dict = data_dict.item()
        # print(data_dict.keys())
        
        with torch.no_grad():
            data_dict = self.voxelizer(data_dict)         
        
        data_dict = self.voxel_3d_generator(data_dict)    
        #-----------生成稀疏张量data_dict['sparse_tensor']----------------[6503,     3]----------[40, 600, 600]---------------------------------------------------------------------------------   
                                              
        # 对稀疏张量进行3D的encoder                               #sparse_tensor:[体素，通道数]   .spatial_shape
    
        
        data_dict = self.conv1a(data_dict)
        # print("conv1a_points",data_dict['points'][:,1].max(),data_dict['points'][:,1].min())
        # print("conv1a_points",data_dict['points'][:,2].max(),data_dict['points'][:,2].min())
        # print("conv1a_points",data_dict['points'][:,0].max(),data_dict['points'][:,0].min())
        # print("conv1a_sparse_tensor",data_dict['sparse_tensor'].features.shape)
        # print("conv1a_sparse_tensor",data_dict['sparse_tensor'].spatial_shape)
        # print("conv1a_sparse_tensor",data_dict['sparse_tensor'].indices)
        # print("conv1a",data_dict['sparse_tensor'].indices.min(),data_dict['sparse_tensor'].indices.max())
        out1b = self.conv1b(data_dict)['sparse_tensor'] 
        # print("conv1b",out1b.features.shape)
        # print("conv1b",out1b.spatial_shape)
        # print("conv1b",out1b.indices)
        # print("conv1b",out1b.indices[:,2].min(),out1b.indices[:,2].max())
        data_dict = self.conv2a(data_dict)                
        # print("conv2a",data_dict['sparse_tensor'].features.shape)
        # print("conv2a",data_dict['sparse_tensor'].spatial_shape)
        # print("conv2a",data_dict['sparse_tensor'].indices)
        # print("conv2a",data_dict['sparse_tensor'].indices[:,2].min(),data_dict['sparse_tensor'].indices[:,2].max())
        out3a = self.conv3a(data_dict)['sparse_tensor'] 
        # print("conv3a",out3a.features.shape)
        # print("conv3a",out3a.spatial_shape)
        # print("conv3a",out3a.indices)
        # print("conv3a",out3a.indices[:,2].min(), out3a.indices[:,2].max())
        data_dict = self.conv3b(data_dict)
        # print("conv3b",data_dict['sparse_tensor'].features.shape)
        # print("conv3b",data_dict['sparse_tensor'].spatial_shape)
        # print("conv3b",data_dict['sparse_tensor'].indices)
        # print("conv3b",data_dict['sparse_tensor'].indices[:,2].min(),data_dict['sparse_tensor'].indices[:,2].max())
        data_dict = self.conv4a(data_dict)     
        # print("conv4a",data_dict['sparse_tensor'].features.shape)
        # print("conv4a",data_dict['sparse_tensor'].spatial_shape)
        # print("conv4a",data_dict['sparse_tensor'].indices)
        # print("conv4a",data_dict['sparse_tensor'].indices[:,2].min(),data_dict['sparse_tensor'].indices[:,2].max())
        data_dict = self.conv4b(data_dict)  
        # print("conv4b",data_dict['sparse_tensor'].features.shape)
        # print("conv4b",data_dict['sparse_tensor'].spatial_shape)
        # print("conv4b",data_dict['sparse_tensor'].indices)
        # print("conv4b",data_dict['sparse_tensor'].indices[:,2].min(),data_dict['sparse_tensor'].indices[:,2].max())
        out = self.conv4c(data_dict)['sparse_tensor']     
        # print("conv4c",out.spatial_shape)
        # print("conv4c",out.indices)
        # print("conv4c",out.indices[:,2].min(),out.indices[:,2].max())
        
        # multi-scale                                  
        data_dict = self.conv5a(self.msf(out3a, out1b, out, data_dict))
        # print("conv5a",data_dict['sparse_tensor'].features.shape)
        # print("conv5a",data_dict['sparse_tensor'].spatial_shape)
        # print("conv5a",data_dict['sparse_tensor'].indices)
        # print("conv5a",data_dict['sparse_tensor'].indices[:,2].min(),data_dict['sparse_tensor'].indices[:,2].max())
        # out5b = self.conv5b(data_dict)['sparse_tensor']
        
        data_dict = self.conv5b(data_dict)     
        # print("conv5b",data_dict['sparse_tensor'].features.shape)
        # print("conv5b",data_dict['sparse_tensor'].spatial_shape)
        # print("conv5b",data_dict['sparse_tensor'].indices)
        # print("conv5b",data_dict['sparse_tensor'].indices[:,2].min(),data_dict['sparse_tensor'].indices[:,2].max())
        
        # loc_feats = data_dict['sparse_tensor'].features  # 定位特征 [N, 3500]
        # print("sparse_tensor",data_dict['sparse_tensor'].features.shape)

        out = self.convout(data_dict)

        # 对比学习通过仅影响检测的方式来引导定位
        loc_sparse_tensor = data_dict["sparse_tensor"]
        loc_sparse_tensor = loc_sparse_tensor.replace_feature(
            loc_sparse_tensor.features.detach().clone()
        )
        loc_data_dict = {
            "sparse_tensor": loc_sparse_tensor,
            "coors": data_dict["coors"],
            "batch_size": data_dict["batch_size"],
        }

        loc_data_dict = self.loc_convbev(loc_data_dict) # loc_bev

        data_dict = self.convbev(data_dict)
        
        #-----------------------------生成2D稀疏特征data_dict['spatial_features']-----------------------------------------------------------------------------------------------------------  
        print("convbev",data_dict['sparse_tensor'].features.shape)
        # print("convbev",data_dict['sparse_tensor'].spatial_shape)
        # print("convbev",data_dict['sparse_tensor'].indices)
        # print("convbev",data_dict['sparse_tensor'].indices[:,2].min(),data_dict['sparse_tensor'].indices[:,2].max())
        # exit()

        # 新分支：特征对齐 -------------------------------------------------
        # 1. 提取稀疏张量并转换为BEV
        sparse_tensor = loc_data_dict['sparse_tensor']
        bev_feature = sparse_tensor.dense()  # [1, 64, 5, 75, 75]
        
        # 2. 高度压缩
        N, C, D, H, W = bev_feature.shape
        bev_feature = bev_feature.view(N, C * D, H, W)  # [1, 320, 75, 75]

        # 3. 通过专用backbone处理
        bev_data_dict = {'spatial_features': bev_feature}
        bev_data_dict = self.loc_bev_backbone(bev_data_dict)  # 关键处理步骤
        aligned_loc_feature = bev_data_dict['spatial_features_2d']

        data_dict['aligned_loc_feature'] = aligned_loc_feature

        data_dict = self.maptobev(data_dict)

        data_dict = self.backbone2d(data_dict)

        print("aligned_loc_feature",data_dict['aligned_loc_feature'].shape)
        print("spatial_features_2d",data_dict['spatial_features_2d'].shape)
        
        # import os

        # output_dir = "/home/ssd1/code/anchor_SG/output_heatmap_visual"
        # os.makedirs(output_dir, exist_ok=True)
        
        # aligned_loc_feature = data_dict['aligned_loc_feature'].cpu().numpy() if torch.is_tensor(data_dict['aligned_loc_feature']) else data_dict['aligned_loc_feature']
        # np.save(os.path.join(output_dir, f'{self.file_counter}_aligned_loc_feature.npy'), aligned_loc_feature)
        # del aligned_loc_feature
        
        # features = data_dict['spatial_features_2d'].cpu().numpy() if torch.is_tensor(data_dict['spatial_features_2d']) else data_dict['spatial_features_2d']
        # np.save(os.path.join(output_dir, f'{self.file_counter}_spatial_features_2d.npy'), features)
        # del features
        
        data_dict = self.AnchorHeadSingle(data_dict) 

        data_dict['aligned_loc_feature'] = data_dict['aligned_loc_feature'].permute(0, 2, 3, 1)
        N, H, W, C = data_dict['aligned_loc_feature'].shape
        # data_dict['aligned_loc_feature'] = data_dict['aligned_loc_feature'].reshape(N * H * W, C)
        # print("data_dict['aligned_loc_feature'].shape: ", N)

        # 获取对比学习所需特征

        data_dict['aligned_det_feature'] = data_dict['spatial_features_2d'].permute(0, 2, 3, 1)
        N, H, W, C = data_dict['aligned_det_feature'].shape
        # data_dict['aligned_det_feature'] = data_dict['aligned_det_feature'].reshape(N * H * W, C)
        # print(N)
        
        # 保存对比学习数据
        data_dict.update({
            'contrast_data': (data_dict['aligned_loc_feature'], data_dict['aligned_det_feature'])
        })
        
        # cls_preds = data_dict['cls_preds'].cpu().numpy() if torch.is_tensor(data_dict['cls_preds']) else data_dict['cls_preds']
        # np.save(os.path.join(output_dir, f'{self.file_counter}_cls_preds.npy'), cls_preds)
        # del cls_preds

        # box_preds = data_dict['box_preds'].cpu().numpy() if torch.is_tensor(data_dict['box_preds']) else data_dict['box_preds']
        # np.save(os.path.join(output_dir, f'{self.file_counter}_box_preds.npy'), box_preds)
        # del box_preds
        
        self.file_counter += 1
        
        return out.features, data_dict
