import os
import numpy as np
import os.path as osp
import h5py
import random
import math
from torch.utils import data
# import transforms3d.quaternions as txq
from util.pose_util import calibrate_process_poses, filter_overflow_ts, qlog
from data.robotcar_sdk.python.velodyne import load_velodyne_binary_seg, load_velodyne_binary, load_velodyne_binary_seg_feature32
import time
from OpenPCDet.pcdet.utils import common_utils
BASE_DIR = osp.dirname(osp.abspath(__file__))
# def mask_points_by_range(points, limit_range):
#     mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
#            & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
#     return mask
def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] > limit_range[0]) & (points[:, 0] < limit_range[3]) \
           & (points[:, 1] > limit_range[1]) & (points[:, 1] < limit_range[4])\
          & (points[:, 2] > limit_range[2]) & (points[:, 2] < limit_range[5])
    return mask
import torch
def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False
def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()

    # print(f"Points shape before rotation: {points[:, :, 0:3].shape}")
    # print(f"Rotation matrix shape: {rot_matrix.shape}")

    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def global_rotation(gt_boxes,random_angle):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    if gt_boxes.ndim == 1:
        gt_boxes = gt_boxes[np.newaxis, :]
    # random_angle = random_angle.item()
    # print("2",random_angle)

    gt_boxes[:, 0:3] = rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([random_angle]))[0]
    gt_boxes[:, 6] += random_angle

    return gt_boxes

class QEOxford(data.Dataset):
    split_filename = None  # 类变量

    def __init__(self, data_path, split='train', real=False, num_grid_x=1, num_grid_y=1, block_num=1, augment=False):
        # directories，data_path='/home/ssd1/data/'
        #初始化数据集路径和分割信息train_split、valid_split
        lidar = 'velodyne_left'
        data_dir = osp.join(data_path, 'Oxford')
        extrinsics_dir = osp.join(BASE_DIR, 'robotcar_sdk', 'extrinsics')

        # 根据 split 选择训练集或验证集的序列
        if split == 'train':
            # split_filename = osp.join(data_dir, 'train_split_1412.txt')
            self.split_filename = osp.join(data_dir, 'train_split.txt')
        else:
            if self.split_filename is None:  # 如果类变量未设置，则用默认值
                self.split_filename = osp.join(data_dir, 'valid_split_17-14.txt')
        #读取分割文件中的序列列表

        with open(self.split_filename, 'r') as f:
            seqs = [l.rstrip() for l in f if not l.startswith('#')]

        # np.random.seed(1000)
        
        # 初始化存储路径和位姿的字典
        ps = {}
        ts = {}
        vo_stats = {}
        pcs_all = []
        self.pcs = []
        # print("I'M QEOxford")
        
        # 遍历每个序列，处理位姿和点云数据
        for seq in seqs:
            # 构造序列目录路径
            seq_dir = osp.join(data_dir, seq + '-radar-oxford-10k')
            # 构造 h5 文件路径
            h5_path = osp.join(seq_dir, lidar + '_calibrate' + str(real) + '.h5')  
            #h5_path='/home/ssd1/data/Oxford/2019-01-15-13-06-37-radar-oxford-10k/velodyne_left_calibrateFalse.h5'
            
            # 如果 h5 文件不存在，则进行插值处理并保存到 h5 文件
            if not os.path.isfile(h5_path):
                print('interpolate ' + seq)
                ts_filename = osp.join(seq_dir, lidar + '.timestamps')
                with open(ts_filename, 'r') as f:
                    ts_raw = [int(l.rstrip().split(' ')[0]) for l in f]
                ins_filename = osp.join(seq_dir, 'gps', 'ins.csv')
                ts[seq] = filter_overflow_ts(ins_filename, ts_raw)
                rot = np.fromfile(osp.join(seq_dir, 'rot_tr.bin'), dtype=np.float32).reshape(-1, 9)
                t = np.fromfile(osp.join(seq_dir, 'tr_add_mean.bin'), dtype=np.float32).reshape(-1, 3)
                ps[seq] = np.concatenate((rot, t), axis=1)  # (n, 12)
                # write to h5 file
                print('write interpolate pose to ' + h5_path)
                h5_file = h5py.File(h5_path, 'w')
                h5_file.create_dataset('valid_timestamps', data=np.asarray(ts[seq], dtype=np.int64))
                h5_file.create_dataset('poses', data=ps[seq])
            # 加载 h5 文件中的位姿数据
            else:
                print("load " + seq + ' pose from ' + h5_path)
                h5_file = h5py.File(h5_path, 'r')
                ts[seq] = h5_file['valid_timestamps'][...]
                ps[seq] = h5_file['poses'][...]
            vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}         #s是缩放因子



           #根据5h文件时间戳ts获取点云文件路径列表：
            pcs_all.extend(
                [osp.join(seq_dir, 'sphere_velodyne_left_feature32', '{:d}.bin'.format(t)) for t in ts[seq]])

        #根据5h文件读取ps或保存位姿归一化信息，对12维pose真值的后三维xyz，求平均值和标准差
        poses = np.empty((0, 12))
        for p in ps.values():
            poses = np.vstack((poses, p))
        pose_stats_filename = osp.join(data_dir, 'QEOxford_pose_stats.txt')
        if split=='train':
            mean_t = np.mean(poses[:, 9:], axis=0)  # (3,)
            std_t = np.std(poses[:, 9:], axis=0)  # (3,)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)

        #将从h5文件中读取到的12维pose转换为平移(3维)+对数四元数(3维)，并进行对齐和归一化
        self.poses = np.empty((0, 6))
        self.rots = np.empty((0, 3, 3))
        self.poses_max = np.empty((0, 2))
        self.poses_min = np.empty((0, 2))
        poses_all = np.empty((0, 6))
        rots_all = np.empty((0, 3, 3))
        pose_max_min_filename = osp.join(data_dir, 'pose_max_min.txt')

        for seq in seqs:
            pss, rotation, pss_max, pss_min = calibrate_process_poses(poses_in=ps[seq], mean_t=mean_t,
                                                                      align_R=vo_stats[seq]['R'],
                                                                      align_t=vo_stats[seq]['t'],
                                                                      align_s=vo_stats[seq]['s'])
            #将pss合并到一起poses_all，将rotation合并到一起rots_all
            poses_all = np.vstack((poses_all, pss))
            self.poses_max = np.vstack((self.poses_max, pss_max))
            self.poses_min = np.vstack((self.poses_min, pss_min))
            rots_all = np.vstack((rots_all, rotation))
        if split == 'train':
            print("============:" + split)
            self.poses_max = np.max(self.poses_max, axis=0) + mean_t[:2]           #计算全局位姿的最大最小x，y坐标的位置，分别是沿x，沿y的最大值
            self.poses_min = np.min(self.poses_min, axis=0) + mean_t[:2]         
            block_size = list((np.array(list(self.poses_max)) - np.array(list(self.poses_min))) / block_num)     #计算区域划分的尺寸大小
            np.savetxt(pose_max_min_filename, np.vstack((self.poses_max, self.poses_min)), fmt='%8.7f')          #将全局坐标系下的最大和最小边界框坐标保存到文件
        else:
            self.poses_max, self.poses_min = np.loadtxt(pose_max_min_filename)
            block_size = list((np.array(list(self.poses_max)) - np.array(list(self.poses_min))) / block_num)
        poses_all_real = poses_all[:, :2] + mean_t[:2] - self.poses_min
        # 根据 block_num 将区域划分为子区域，并筛选点云数据divide the area into subregions
        if block_num != 1:
            for i in range(len(poses_all)):
                if int((poses_all_real[i, 0]) / block_size[0]) == num_grid_x and int(
                        poses_all_real[i, 1] / block_size[1]) == num_grid_y:
                    self.poses = np.vstack((self.poses, poses_all[i]))
                    self.pcs.append(pcs_all[i])
                    self.rots = np.vstack((self.rots, rots_all[i].reshape(1, 3, 3)))

        else:
            self.poses = poses_all
            print("poses_all:" + str(len(self.poses)))   
            self.pcs = pcs_all
            self.rots = rots_all
        
        # 设置数据集的split类型
        self.split = split
        print("============:" + self.split)
        if split == 'train':
            print("train data num:" + str(len(self.poses)))
        else:    
            print("valid data num:" + str(len(self.poses)))

        # 统计总帧数和类别分布
        total_frames = 0
        class_distribution = {}
        
        # 遍历所有序列统计总帧数
        for seq in seqs:
            seq_dir = osp.join(data_dir, seq + '-radar-oxford-10k')
            ts_filename = osp.join(seq_dir, lidar + '.timestamps')
            with open(ts_filename, 'r') as f:
                total_frames += len(f.readlines())
        
        # 统计类别分布
        for scan_path in self.pcs:
            base_path = os.path.dirname(os.path.dirname(scan_path))
            label_path_bbox = os.path.join(base_path, "labeltxt" + f"/{scan_path[-20:-4]}.txt")
            try:
                with open(label_path_bbox) as f:
                    for line in f.readlines():
                        numbers = list(map(float, line.split()))
                        if numbers:
                            class_id = int(numbers[-1])
                            class_distribution[class_id] = class_distribution.get(class_id, 0) + 1
            except FileNotFoundError:
                continue

        print("\nDataset Statistics:")
        print(f"Total frames in dataset: {total_frames}")
        print(f"Key frames used for detection: {len(self.poses)}")
        print("\nClass distribution in key frames:")
        for class_id, count in sorted(class_distribution.items()):
            print(f"Class {class_id}: {count} instances")

    def __getitem__(self, index):
        ## 获取索引对应的点云文件路径
        scan_path = self.pcs[index]
        # print("scan_path" + scan_path)
        #读取点云文件，reshape为(N,35)，这里假设每个点有35个特征
        ptcld = load_velodyne_binary_seg_feature32(scan_path).reshape(-1, 35)
        # mask =mask_points_by_range(ptcld, [-75.2, -75.2, -2, 75.2, 75.2, 4])
        # mask = mask_points_by_range(ptcld, [-59.9, -59.9, -2, 59.9, 59.9, 4])
        # mask = mask_points_by_range(ptcld, [-60, -60, -2, 60, 60, 6])
        # ptcld=ptcld[mask]
        xyz = ptcld[:, :3]
        xyz[:,[2]]+=1.69
        # feature = ptcld[:, 3:]
        
        #读取bbox
        base_path = os.path.dirname(os.path.dirname(scan_path))
        
        # 训练读取动静标签
        if self.split == 'train':
            label_path_bbox = os.path.join(base_path, "labeltxt" + f"/{scan_path[-20:-4]}.txt")
            label_path_bbox_m = os.path.join(base_path, "label_m" + f"/{scan_path[-20:-4]}.txt")
            boxes = []
            boxes_m = []
            with open(label_path_bbox) as f:
                for line in f.readlines():
                    numbers = list(map(float, line.split()))
                    if numbers:
                        boxes.append(numbers)
                    # else:
                    #     # 当 numbers 为空时，添加一个形状为 (8,) 的空数组
                    #     boxes.append(np.empty(8))
            # 在所有数据读取完毕后，将列表转换为 NumPy 数组
            boxes = np.array(boxes)
            if len(boxes) == 0:
                # boxes = np.empty((0,8))
                boxes = np.empty((0,9))

            if boxes.shape[1] > 0:  # 确保至少有一个维度
                boxes[:, -1] = boxes[:, -1].astype(int)  # 将最后一个类别维度转换为整数

            with open(label_path_bbox_m) as f:
                for line in f.readlines():
                    numbers = list(map(float, line.split()))
                    if numbers:
                        boxes_m.append(numbers)
                    # else:
                    #     # 当 numbers 为空时，添加一个形状为 (8,) 的空数组
                    #     boxes.append(np.empty(8))
            # 在所有数据读取完毕后，将列表转换为 NumPy 数组
            boxes_m = np.array(boxes_m)
            if len(boxes_m) == 0:
                boxes_m = np.empty((0,9))

            # 以 boxes 的行数为准，合并 boxes_m 的最后一列
            if boxes.shape[0] > 0:
                if boxes_m.shape[0] >= boxes.shape[0]:
                    # 如果 boxes_m 行数 >= boxes，取前 boxes.shape[0] 行
                    last_column_m = boxes_m[:boxes.shape[0], -1].reshape(-1, 1)
                else:
                    # 如果 boxes_m 行数 < boxes，填充默认值（例如 0 或 np.nan）
                    default_value = 0  # 可以改成 np.nan 或其他默认值
                    last_column_m = np.full((boxes.shape[0], 1), default_value)
                    last_column_m[:boxes_m.shape[0]] = boxes_m[:, -1].reshape(-1, 1)
                
                # 合并到 boxes
                boxes = np.hstack((boxes, last_column_m))
        # 测试
        else:
            label_path_bbox = os.path.join(base_path, "labeltxt" + f"/{scan_path[-20:-4]}.txt")
            boxes = []
            with open(label_path_bbox) as f:
                for line in f.readlines():
                    numbers = list(map(float, line.split()))
                    if numbers:
                        boxes.append(numbers)
                        
            # 在所有数据读取完毕后，将列表转换为 NumPy 数组
            boxes = np.array(boxes)
            if len(boxes) == 0:
                boxes = np.empty((0,8))

            if boxes.shape[1] > 0:  # 确保至少有一个维度
                boxes[:, -1] = boxes[:, -1].astype(int)  # 将最后一个类别维度转换为整数
        
        pose = self.poses[index]  # (6,) 世界坐标系下的xyz [0:3]
        rot = self.rots[index]  # 3*3 世界坐标的旋转矩阵
                
        # pred_scores_file = os.path.join("/home/ssd1/xly/try/code/output_visual", f"{scan_path[-20:-4]}_points.txt")
        # np.savetxt(pred_scores_file,  xyz, fmt='%.6f')

        # pred_scores_file = os.path.join("/home/ssd1/xly/try/code/output_visual", f"{scan_path[-20:-4]}_boxes.txt")
        # np.savetxt(pred_scores_file,   boxes, fmt='%.6f')
        
        # pred_scores_file = os.path.join("/home/ssd1/xly/try/code/output_visual", f"{scan_path[-20:-4]}_pose.txt")
        # np.savetxt(pred_scores_file, pose, fmt='%.6f')   
        
        # pred_scores_file = os.path.join("/home/ssd1/xly/try/code/output_visual", f"{scan_path[-20:-4]}_rot.txt")
        # np.savetxt(pred_scores_file, rot, fmt='%.6f')   
           
        '''定位 和 检测 的增强'''
        if self.split=='train' : # 关闭 定位 和 检测 的增强
            #构造原始位姿矩阵
            orin_pose = np.eye(4)
            orin_pose[0:3,0:3] = rot
            orin_pose[0:3,-1] = pose[:3]
            
            #生成随机平移和旋转
            randoms = np.random.uniform(-2, 2, size=(3,))

            # random_angle = np.random.uniform (-np.pi/4, np.pi/4, size=(1,))
 
            # rotation_matrix = self.rotation_matrix_z(random_angle[0])
            randoms[2] = 0
            random_pose = np.eye(4)
            # random_pose[0:3,0:3] = rotation_matrix
            random_pose[0:3,-1] = randoms
             
            #计算新的全局位姿
            random_pose_1 = np.linalg.inv(random_pose)
            new_global_pose = orin_pose @ random_pose_1
            
            #应用随机 变换到点云
            new_orin_xyz = np.ones(shape= (xyz.shape[0],4))
            new_orin_xyz[:,0:3] = xyz[:,0:3]
            new_local_pts = (random_pose@new_orin_xyz.T).T

            xyz = new_local_pts[:,0:3]
            ptcld[:,:3] = new_local_pts[:,0:3]
            #更新位姿和旋转矩阵

            pose[:3] = new_global_pose[0:3,-1]
            rot = new_global_pose[0:3,0:3]

            #gt_bboxs的旋转增强
            # boxes = global_rotation(boxes, random_angle[0])
            

            #gt_bboxs的平移增强
            boxes[:,:3] = boxes[:,:3] + randoms
            # mask_b = mask_points_by_range(boxes[:,0:3], [-59.9, -59.9, -2, 59.9, 59.9, 4])
            
            # boxes = boxes[mask_b]            
        boxes[:,2]= boxes[:,2] + 1.69
            
        #原始点云mask
        mask1 = mask_points_by_range(xyz, [-59.9, -59.9, -2, 59.9, 59.9, 5.9])
        # mask1 = mask_points_by_range(xyz, [-60, -60, -2, 60, 60, 6])
        xyz = xyz[mask1]
        #feature的语义mask           
        ptcld = ptcld[mask1]
        feature = ptcld[:,3:]
        #bbox的mask
        mask2 = mask_points_by_range(boxes, [-59.9, -59.9, -2, 59.9, 59.9, 5.9])
        # mask2 = mask_points_by_range(boxes, [-60, -60, -2, 60, 60, 6])
        boxes = boxes[mask2]

        #计算ground truth，即点云数据通过位姿变换后的位置
        gt = (rot @ xyz.transpose(1, 0)).transpose(1, 0) + pose[:3].reshape(1, 3)
        
        # pred_scores_file = os.path.join("/home/ssd1/code/anchor_SG/output_visual", f"{scan_path[-20:-4]}_pointcloud.txt")
        # np.savetxt(pred_scores_file, xyz, fmt='%.6f')
        
        # pred_scores_file = os.path.join("/home/ssd1/xly/try/code/output_visual", f"{scan_path[-20:-4]}_new_boxes.txt")
        # np.savetxt(pred_scores_file,   boxes, fmt='%.6f')
        
        # pred_scores_file = os.path.join("/home/ssd1/xly/try/code/output_visual", f"{scan_path[-20:-4]}_gt.txt")
        # np.savetxt(pred_scores_file,  gt, fmt='%.6f')
        
        # pred_scores_file = os.path.join("/home/ssd1/xly/try/code/output_visual", f"{scan_path[-20:-4]}_new_pose.txt")
        # np.savetxt(pred_scores_file, new_global_pose, fmt='%.6f')  
        
        # pred_scores_file = os.path.join("/home/ssd1/xly/try/code/output_visual", f"{scan_path[-20:-4]}_feature.txt")
        # np.savetxt(pred_scores_file, feature, fmt='%.6f')  
         
        #根据split类型，构造不同的标签
        if self.split=='train':
            labels = np.concatenate((xyz, gt, feature), axis=1)              
            
        else: 
            labels = np.concatenate((xyz, gt), axis=1)
        #创建数据字典，存储处理后的点云数据和标签
        data_dict = {}
        data_dict['xyz'] = xyz
        data_dict['labels'] = labels
        data_dict['pose'] = pose
        data_dict['rot'] = rot
        boxes[:, -1] = boxes[:, -1].astype(int)
        data_dict['gt_boxes'] = boxes
        
        return data_dict
        
    def min_max_normalization(self,features):
        min_value = 0
        max_value = 1
        normalized_array = np.interp(features, (features.min(), features.max()), (min_value, max_value))
        return normalized_array
    def min_max(self,x):
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        return x
    def rotation_matrix_z(self,theta ):
  
        Rz = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        return Rz
    def __len__(self):
        return len(self.poses)