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
    def __init__(self):
        """

        """
        super().__init__()
        self.point_cloud_range = [-59.9, -59.9, -2, 59.9, 59.9, 5.9]
    
    def create_bev_mask(self, bev_map, box):
        bounding = self.point_cloud_range
        begin_w = bounding[3] - bounding[0]
        begin_h = bounding[4] - bounding[1]
        x, y, z, h, w, l, yaw = box[:7]
        _, H, W = bev_map.shape
        bev_map_ = F.normalize(bev_map, dim=0, p=2)
        center_x = ((x-bounding[0]) / begin_w) * W
        center_y = ((y-bounding[1]) / begin_h) * H
        return bev_map_[:, int(center_y), int(center_x)]
    
    def forward(self, data_dict):
        '''
        spatial_features_2d: 
        '''
        spatial_features_2d = data_dict['spatial_features_2d']
        gt_boxes = data_dict["gt_boxes"]
        contrastive_loss = 0
        # print(data_dict["gt_boxes"].shape)
        for ba in range(spatial_features_2d.shape[0]): # batch_size
            static_det_feature_list = []
            dynamic_det_feature_list = []
            # print(data_dict["gt_boxes"][ba, :, :].shape)
            for num_gt in range(data_dict["gt_boxes"][ba, :, :].shape[0]): 
                # if object_bbx_center[ba, num_gt, 0] != 0:
                if gt_boxes[ba, num_gt, :][-1] == 0:
                    det_feature = self.create_bev_mask(spatial_features_2d[ba, :, :, :], gt_boxes[ba, num_gt, :])
                    # print(det_feature.reshape(1, -1).shape)
                    static_det_feature_list.append(det_feature.reshape(1, -1))
                if gt_boxes[ba, num_gt, :][-1] == 1:
                    det_feature = self.create_bev_mask(spatial_features_2d[ba, :, :, :], gt_boxes[ba, num_gt, :])
                    # print(det_feature.reshape(1, -1).shape)
                    dynamic_det_feature_list.append(det_feature.reshape(1, -1))
                
        # print(len(static_det_feature_list))
        # print(len(dynamic_det_feature_list))
        if len(static_det_feature_list) > 0 and len(dynamic_det_feature_list) > 0:
            static_det_feature = torch.cat(static_det_feature_list, dim=0).reshape(-1, 128) # [M, 128]
            dynamic_det_feature = torch.cat(dynamic_det_feature_list, dim=0).reshape(-1, 128) # [N, 128]
            # 计算对比损失
            self.contrastive_loss = ContrastiveLoss()
            contrastive_loss = self.contrastive_loss(static_det_feature, dynamic_det_feature) * 0.01
        print("det_contrastive_loss: ", contrastive_loss)
        return contrastive_loss

class LocContrastiveLoss(nn.Module):
    def __init__(self, topk=10, threshold=0.5, margin=0.5):
        super().__init__()
        self.point_cloud_range = [-59.9, -59.9, -2, 59.9, 59.9, 5.9]
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

    def extract_feature_at_box(self, feature_map, box):
        """从 GT box 中心提取检测特征"""
        bounding = self.point_cloud_range
        begin_w = bounding[3] - bounding[0]
        begin_h = bounding[4] - bounding[1]
        x, y = box[0], box[1]
        _, H, W = feature_map.shape
        center_x = ((x - bounding[0]) / begin_w) * W
        center_y = ((y - bounding[1]) / begin_h) * H
        return feature_map[:, int(center_y), int(center_x)]

    def forward(self, data_dict):
        loc_features, det_features = data_dict['contrast_data']
        gt_boxes = data_dict["gt_boxes"]
        contrastive_loss = 0

        for ba in range(loc_features.shape[0]):  # batch_size
            # (1) 提取检测特征（基于 GT boxes）
            static_det_feature_list = []
            for num_gt in range(gt_boxes[ba].shape[0]):
                if gt_boxes[ba, num_gt, -1] == 0:  # 有效 GT box
                    det_feature = self.extract_feature_at_box(det_features[ba], gt_boxes[ba, num_gt])
                    static_det_feature_list.append(det_feature.unsqueeze(0))
            
            if not static_det_feature_list:
                continue  # 如果没有有效 GT box，跳过该样本
            
            static_det_features = torch.cat(static_det_feature_list, dim=0)  # (N_gt, C)

            # (2) 提取定位特征（基于 loc_features 的高亮点）
            peaks = self.find_peaks(loc_features[ba])
            loc_features_batch = self.extract_feature_at_peaks(loc_features[ba], peaks)  # (N_peaks, C)

            if loc_features_batch.shape[0] == 0:
                continue  # 如果没有高亮点，跳过

            # (3) 计算对比损失（鼓励检测和定位特征拉远）
            similarity = F.cosine_similarity(
                static_det_features.unsqueeze(1),  # (N_gt, 1, C)
                loc_features_batch.unsqueeze(0),    # (1, N_peaks, C)
                dim=-1
            )
            # 希望相似度 < margin（鼓励不相关）
            loss = F.relu(similarity - self.margin).mean()
            contrastive_loss += loss

        return contrastive_loss / loc_features.shape[0]  # 平均 batch loss

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
