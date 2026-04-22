# pylint: disable=no-member
import argparse
import os
import sys
import numpy as np
import random
import time
import torch
import copy
import matplotlib
# Oxford
from data.oxford import Oxford
# QEOxford
from data.qeoxford import QEOxford
from copy import deepcopy
# NCLT
from data.nclt import NCLT
from util import config
from model.lisa import LiSA
from model.sc2pcr import Matcher
from data.dataset import sgloc_data, collate_fn_default
from util.pose_util import val_translation, val_rotation, qexp
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.utils.data import DataLoader
from os import path as osp
from tqdm import tqdm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data.distributed import DistributedSampler
from scipy.spatial import cKDTree
# from torchstat import stat
import pickle
from OpenPCDet.pcdet.models.detectors.detector3d_template import Detector3DTemplate

#配置解析
def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Localization')
    parser.add_argument('--config', type=str, default='config/QEOxford/sgloc.yaml',help='config file')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()


def main(args, start_epoch=0, end_epoch=1):
    global TOTAL_ITERATIONS
    TOTAL_ITERATIONS = 0
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    # for (k, v) in args.items():
    #     print('%s: %s' % (str(k), str(v)))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise ValueError("GPU not found!")

    if args.data_name == 'Oxford':
        test_pt_dataset = Oxford(args.data_root,
                                 split='test')
    elif args.data_name == 'QEOxford':
        test_pt_dataset = QEOxford(args.data_root,
                                  split='test',augment=False)
    elif args.data_name == 'NCLT':
        test_pt_dataset = NCLT(args.data_root,
                               split='test')
    else:
        print('Only supprot Oxford, QEOxford and NCLT')

    lenset = len(test_pt_dataset)
    dataset = sgloc_data(test_pt_dataset, args)
    # tran_sampler = DistributedSampler(dataset)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=8,  # 增加到 8 或更多，根据 CPU 核心数调整
        pin_memory=True,
        prefetch_factor=2,  # 预取因子
        persistent_workers=True,  # 保持 worker 进程存活
        collate_fn=collate_fn_default
    )
    test_writer = SummaryWriter(os.path.join(args.save_path, 'test'))
    
    #model的输入args就是lisa.yaml中参数
    model = LiSA(args)
    model = model.to(device)
    params = sum([x.nelement() for x in model.parameters()]) / 1e6
    print('#Model parameters: %.4f' % params)
    
    ransac = Matcher(inlier_threshold=args.threshold)

    LOG_FOUT.write("\n")
    LOG_FOUT.flush()
    for epoch in range(start_epoch, end_epoch):
        log_string('**** EPOCH %03d ****' % epoch)
        resume_filename = args.resume + str(epoch) + '.tar'
        print("Resuming From ", resume_filename)
        checkpoint = torch.load(resume_filename, 'cuda:0')
        saved_state_dict = checkpoint['state_dict']
        model.load_state_dict(saved_state_dict,strict=False)
        sys.stdout.flush()
        valid_one_epoch(model, ransac, test_loader, test_writer, lenset, device, args.threshold)
        torch.cuda.empty_cache()

def valid_one_epoch(model, ransac, test_loader, test_writer, lenset, device, threshold):
    import time
    gt_translation = np.zeros((lenset, 3))
    pred_translation = np.zeros((lenset, 3))
    gt_rotation = np.zeros((lenset, 4))
    pred_rotation = np.zeros((lenset, 4))
    pred_matrix=np.eye(4)

    error_t = np.zeros(lenset)
    error_txy = np.zeros(lenset)
    error_q = np.zeros(lenset)

    time_results_network = np.zeros(lenset)
    time_results_ransac = np.zeros(lenset)

    total_data_time = 0
    total_infer_time = 0
    total_ransac_time = 0
    total_step_time = 0

    tqdm_loader = tqdm(test_loader, total=len(test_loader))
    last_time = time.time()
    
    class_name=[ 'Car', 'Pedestrian', 'Cyclist' ]
    # class_name=[ 'Vehicle', 'Pedestrian', 'Cyclist' ]
    
    # def waymo_eval(eval_det_annos, eval_gt_annos):
    #     from  OpenPCDet.pcdet.datasets.waymo.waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
    #     eval = OpenPCDetWaymoDetectionMetricsEstimator()

    #     ap_result_str = eval.waymo_evaluation(
    #         eval_det_annos, eval_gt_annos, class_name=class_name,
    #         distance_thresh=1000, fake_gt_infos=False
    #     )
    #     return ap_result_str
    
    # def once_eval(eval_det_annos, eval_gt_annos):
    #     from OpenPCDet.pcdet.datasets.once.once_eval.evaluation import get_evaluation_results
        
    #     ap_result_str,ap_dict = get_evaluation_results(eval_gt_annos, eval_det_annos, classes=class_name )
    #     return ap_result_str,ap_dict    

    annos_gt_list = []
    annos_det_list = []
    global_annos_gt_list = []
    global_annos_det_list = []
    
    kk = 0
    a = Detector3DTemplate(model_cfg=args, num_class=3,dataset=tqdm_loader)
    
    for step, input_dict in enumerate(tqdm_loader):
        step_start = time.time()
        # 1. 数据加载时间
        data_time = step_start - last_time
        total_data_time += data_time

        # 2. 模型推理时间
        infer_start = time.time()
        val_pose = input_dict['pose'].reshape(-1, 6)
        batch_size = val_pose.size(0)
        pred_t = np.zeros((batch_size, 3))
        pred_q = np.zeros((batch_size, 4))
        pred_trans=np.zeros([batch_size, 4,4])
        index_list = [0] # 用于存放索引
        start_idx = step * args.batch_size_val
        end_idx = min((step + 1) * args.batch_size_val, lenset)
        # 在得到gt时，仅减去了均值
        gt_translation[start_idx:end_idx, :] = val_pose[:, :3].numpy() + pose_m
        gt_rotation[start_idx:end_idx, :] = np.asarray([qexp(q) for q in val_pose[:, 3:].numpy()])
        # 需要GPU计算的数据
        input_dict['points'] = input_dict['points'].to(device, dtype=torch.float32)
        input_dict['batch_idx'] = input_dict['batch_idx'].to(device, dtype=torch.int8)
        input_dict['labels'] = input_dict['labels'].to(device, dtype=torch.float32)

        orin_data = input_dict['points'].clone().cpu().numpy()
        orin_batch_idx = input_dict['batch_idx'].clone().cpu().numpy()

        pred_shift,  data_dict = run_model(model, input_dict, True)
        infer_end = time.time()
        infer_time = infer_end - infer_start
        total_infer_time += infer_time

        # 3. RANSAC/后处理时间
        ransac_start = time.time()
        labels, coors, det_gt = data_dict['labels'], data_dict['coors'],  data_dict['gt_boxes']
        det_results,_= a.post_processing(batch_dict=data_dict)
        pose, rot = data_dict['pose'], data_dict['rot']

        sup_point = labels[:, :3]
        pred_point = sup_point + pred_shift
        for i in range(batch_size):
            index_list.append(index_list[i] + torch.sum(coors[:, 0]==i))
        gt_point = sup_point
          
        def density(points, orin_pts, rad=3):
            trees = cKDTree(orin_pts[:, 0:3])
            num = trees.query_ball_point(
                points[:, 0:3], r=rad,
                return_length=True)
            return num

        for bt_id in range(batch_size):

            this_sup = gt_point[index_list[bt_id]:index_list[bt_id+ 1], :]
            this_pred = pred_point[index_list[bt_id]:index_list[bt_id + 1], :]

            
            orin_pts = orin_data[orin_batch_idx==i]
            
            new_sup = torch.clone(this_sup)
            new_sup = new_sup.cpu().numpy()
            this_density = density(new_sup, new_sup)
            nargs = np.argsort(-this_density)
            nargs = torch.from_numpy(nargs).cuda()
            min_l = 700
            if len(this_density)<=700:
                min_l = len(this_density)

            this_sup = this_sup[nargs[0:min_l]]
            this_pred = this_pred[nargs[0:min_l]]
            
            start = time.time()
            batch_pred_t, batch_pred_q, batch_pred_trans = ransac.estimator(this_sup.unsqueeze(0), this_pred.unsqueeze(0))
            end = time.time()
            cost_time = (end - start) / batch_size
            time_results_ransac[start_idx:end_idx] = [cost_time for _ in range(batch_size)]

            pred_t[bt_id, :] = batch_pred_t
            pred_q[bt_id, :] = batch_pred_q
            pred_trans[bt_id, :] = batch_pred_trans.cpu().numpy()
            
            
            cur_boxes = det_results[bt_id]['pred_boxes'].cpu().numpy()
            cur_scores = det_results[bt_id]['pred_scores'].cpu().numpy()
            cur_labels = det_results[bt_id]['pred_labels'].cpu().numpy()
            cur_gt = data_dict['gt_boxes'][bt_id][:,:-1]
            
            # 打印检测统计信息
            # print(f"\nDetection statistics for batch {step}:")
            # print(f"Total predictions: {len(cur_boxes)}")
            # print(f"Predictions after threshold: {len(cur_boxes[cur_scores > 0.1])}")
            # print(f"Class distribution: {np.bincount(cur_labels[cur_scores > 0.1].astype(int))}")
            
            # 降低检测阈值
            mask = cur_scores > 0.1  # 降低阈值到 0.1
            cur_boxes = cur_boxes[mask]
            cur_scores = cur_scores[mask]
            cur_labels = cur_labels[mask]
            
            k = cur_gt.__len__() - 1
            while k >= 0 and cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1]
            cur_gt = np.empty((0, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt
            
            cur_gt_label = data_dict['gt_boxes'][bt_id][:,-1]
            cur_gt_label = cur_gt_label[:k+1]
            cur_gt_label =np.empty((0,)) if len(cur_gt_label) == 0 else cur_gt_label
            
            global_boxes=copy.deepcopy(cur_boxes)
            cur_xyz=global_boxes[:,:3]
            new_cur_xyz = np.ones(shape= (cur_xyz.shape[0],4))
            new_cur_xyz[:,0:3] = cur_xyz[:,0:3]
            global_xyz=(pred_trans[bt_id]@new_cur_xyz.T).T
            global_boxes[:,:3]=global_xyz[:,0:3]
            
            # print("cur_boxes##############",cur_gt)
            # print("global_boxes##############",global_boxes)
            # exit()
            global_cur_gt=copy.deepcopy(cur_gt)

            gt_xyz=global_cur_gt[:,:3]
            rot=data_dict['rot'].cpu().numpy().reshape(batch_size,3,3)[bt_id]
            pose=data_dict['pose'].cpu().numpy().reshape(batch_size,6)[bt_id]
            global_gt_xyz=(rot @ gt_xyz.transpose(1, 0)).transpose(1, 0) + pose[:3].reshape(1, 3)
            global_cur_gt[:,:3]=global_gt_xyz

            anno_gt = {
                'name': np.array(class_name)[cur_gt_label.astype(int)-1],
                'difficulty': np.ones_like(cur_gt_label).astype(int),
                'boxes_3d': cur_gt,
                'num_points_in_gt': np.ones_like(cur_gt_label).astype(int)+100
            }
            
            anno_det = {
                'name': np.array(class_name)[cur_labels.astype(int)-1],
                'score': cur_scores,
                'boxes_3d': cur_boxes,
                'frame_id': str(kk)
            }
            
            global_anno_gt = {
                'name': np.array(class_name)[cur_gt_label.astype(int)-1],
                'difficulty': np.ones_like(cur_gt_label).astype(int),
                'boxes_3d': global_cur_gt,
                'num_points_in_gt': np.ones_like(cur_gt_label).astype(int)+100
            }
            
            global_anno_det = {
                'name': np.array(class_name)[cur_labels.astype(int)-1],
                'score': cur_scores,
                'boxes_3d': global_boxes,
                'frame_id': str(kk)
            }
            
            annos_det_list.append(anno_det)
            annos_gt_list.append(anno_gt)
            
            global_annos_det_list.append(global_anno_det)
            global_annos_gt_list.append(global_anno_gt)
            
            kk = kk + 1  

        pred_translation[start_idx:end_idx, :] = pred_t + pose_m
        pred_rotation[start_idx:end_idx, :] = pred_q
        
        
        error_t[start_idx:end_idx] = np.asarray([val_translation(p, q) for p, q in
                                                 zip(pred_translation[start_idx:end_idx, :],
                                                     gt_translation[start_idx:end_idx, :])])
        error_txy[start_idx:end_idx] = np.asarray([val_translation(p, q) for p, q in
                                                 zip(pred_translation[start_idx:end_idx, :2],
                                                     gt_translation[start_idx:end_idx, :2])])

        error_q[start_idx:end_idx] = np.asarray([val_rotation(p, q) for p, q in zip(pred_rotation[start_idx:end_idx, :],
                                                                                    gt_rotation[start_idx:end_idx, :])])

        log_string('MeanXYZTE(m): %f' % np.mean(error_t[start_idx:end_idx], axis=0))
        log_string('MeanXYTE(m): %f' % np.mean(error_txy[start_idx:end_idx], axis=0))
        log_string('MeanRE(degrees): %f' % np.mean(error_q[start_idx:end_idx], axis=0))
        log_string('MedianTE(m): %f' % np.median(error_t[start_idx:end_idx], axis=0))
        log_string('MedianRE(degrees): %f' % np.median(error_q[start_idx:end_idx], axis=0))
        log_string('NetCostTime(s): %f' % np.mean(time_results_network[start_idx:end_idx], axis=0))
        log_string('RANSACCostTime(s): %f' % np.mean(time_results_ransac[start_idx:end_idx], axis=0))
    
    
    #生成pkl的检测结果
    
    with open(osp.join(args.save_path, 'anno_gt_78.pkl'), 'wb') as f:
        pickle.dump(annos_gt_list, f)
    with open(osp.join(args.save_path,'anno_det_78.pkl'), 'wb') as f:
        pickle.dump(annos_det_list, f)
    # with open(osp.join(args.save_path,'global_anno_gt.pkl'), 'wb') as f:
    #     pickle.dump(global_annos_gt_list, f)
    # with open(osp.join(args.save_path,'global_anno_det.pkl'), 'wb') as f:
    #     pickle.dump(global_annos_det_list, f)
    
  
    # ap_result_str = waymo_eval(annos_det_list, annos_gt_list)
    # local_ap_result_str,ret_dict = once_eval(annos_det_list, annos_gt_list)
    # global_ap_result_str,ret_dict = once_eval(global_annos_det_list, global_annos_gt_list)
    # print(ap_result_str)

    mean_ATE = np.mean(error_t)
    mean_xyATE = np.mean(error_txy)
    mean_ARE = np.mean(error_q)
    median_ATE = np.median(error_t)
    median_xyATE = np.median(error_txy)
    median_ARE = np.median(error_q)
    mean_time_network = np.mean(time_results_network)
    mean_time_ransac = np.mean(time_results_ransac)
    
    # log_string('AP_result: %s' % ap_result_str)
    # log_string('AP_result: %s' % local_ap_result_str)
    # log_string('AP_result: %s' % global_ap_result_str)
    log_string('Mean Position Error(m): %f' % mean_ATE)
    log_string('Mean XY Position Error(m): %f' % mean_xyATE)
    log_string('Mean Orientation Error(degrees): %f' % mean_ARE)
    log_string('Median Position Error(m): %f' % median_ATE)
    log_string('Median XY Position Error(m): %f' % median_xyATE)
    log_string('Median Orientation Error(degrees): %f' % median_ARE)
    log_string('Mean Network Cost Time(s): %f' % mean_time_network)
    log_string('Mean Ransac Cost Time(s): %f' % mean_time_ransac)
    test_writer.add_scalar('MeanATE', mean_ATE, TOTAL_ITERATIONS)
    test_writer.add_scalar('MeanARE', mean_ARE, TOTAL_ITERATIONS)

    # save error
    error_t_filename = osp.join(args.save_path, 'error_t.txt')
    error_q_filename = osp.join(args.save_path, 'error_q.txt')
    np.savetxt(error_t_filename, error_t, fmt='%8.7f')
    np.savetxt(error_q_filename, error_q, fmt='%8.7f')

    # trajectory
    
    fig = plt.figure()
    real_pose = pred_translation - pose_m
    gt_pose = gt_translation - pose_m
    plt.scatter(gt_pose[:, 1], gt_pose[:, 0], s=1, c='black')
    plt.scatter(real_pose[:, 1], real_pose[:, 0], s=1, c='red')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.plot(gt_pose[0, 1], gt_pose[0, 0], 'y*', markersize=10)
    image_filename = os.path.join(os.path.expanduser(args.save_path), '{:s}.png'.format('trajectory_' + seq + '_' + str(threshold)))
    fig.savefig(image_filename, dpi=200, bbox_inches='tight')

    # translation_distribution
    fig = plt.figure()
    t_num = np.arange(len(error_t))
    plt.scatter(t_num, error_t, s=1, c='red')
    plt.xlabel('Data Num')
    plt.ylabel('Error (m)')
    # 设置纵轴大小
    plt.ylim(0, 40)
    image_filename = os.path.join(os.path.expanduser(args.save_path), '{:s}.png'.format('distribution_t_' + seq + '_' + str(threshold)))
    fig.savefig(image_filename, dpi=200, bbox_inches='tight')

    # rotation_distribution
    fig = plt.figure()
    q_num = np.arange(len(error_q))
    plt.scatter(q_num, error_q, s=1, c='blue')
    plt.xlabel('Data Num')
    plt.ylabel('Error (degree)')
    # 设置纵轴大小
    plt.ylim(0, 40)
    image_filename = os.path.join(os.path.expanduser(args.save_path), '{:s}.png'.format('distribution_q_' + seq + '_' + str(threshold)))
    fig.savefig(image_filename, dpi=200, bbox_inches='tight')
    

    # save error and trajectory
    error_t_filename = osp.join(args.save_path, 'error_t.txt')
    error_q_filename = osp.join(args.save_path, 'error_q.txt')
    pred_q_filename = osp.join(args.save_path, 'pred_q.txt')
    pred_t_filename = osp.join(args.save_path, 'pred_t.txt')
    gt_t_filename = osp.join(args.save_path, 'gt_t.txt')
    np.savetxt(error_t_filename, error_t, fmt='%8.7f')
    np.savetxt(error_q_filename, error_q, fmt='%8.7f')
    np.savetxt(pred_t_filename, real_pose, fmt='%8.7f')
    np.savetxt(pred_q_filename, pred_rotation, fmt='%8.7f')
    np.savetxt(gt_t_filename, gt_pose, fmt='%8.7f')

    # 4. 总step时间
    step_end = time.time()
    step_time = step_end - step_start
    total_step_time += step_time

    print(f"\n==== 测试计时统计 ====")
    print(f"数据加载总时间: {total_data_time:.3f}s, 平均每步: {total_data_time/len(test_loader):.3f}s")
    print(f"模型推理总时间: {total_infer_time:.3f}s, 平均每步: {total_infer_time/len(test_loader):.3f}s")
    print(f"RANSAC总时间: {total_ransac_time:.3f}s, 平均每步: {total_ransac_time/len(test_loader):.3f}s")
    print(f"总step时间: {total_step_time:.3f}s, 平均每步: {total_step_time/len(test_loader):.3f}s")

def run_model(model, x1, validate=False):
    if not validate:
        model.train()
        return model(x1)
    else:
        with torch.no_grad():
            model.eval()
            return model(x1)

if __name__ == "__main__":
    import gc
    gc.collect()
    args = get_parser()
    
    # test_sequences = ["17-14","17-13","15-13","18-14"]
    test_sequences = ["17-14"]
    start_epoch = 78
    end_epoch = 79
    
    for seq in test_sequences:
        print(f"\n{'='*50}\nTesting sequence: {seq}\n{'='*50}\n")
        
        # 设置数据集和保存路径
        QEOxford.split_filename = os.path.join(args.data_root, 'Oxford', f'valid_split_{seq}.txt')
        print("QEOxford.split_filename:", QEOxford.split_filename)
        args.save_path = os.path.join(args.save_path_test, f'seq{seq}')
        print("Save path:", args.save_path)

        os.makedirs(args.save_path, exist_ok=True)
        
        # 初始化日志
        log_filename = f'log_78_1.txt'
        LOG_FOUT = open(os.path.join(args.save_path, log_filename), 'w')
        LOG_FOUT.write(str(args) + '\n')
        
        # 加载位姿统计数据
        pose_stats_file = os.path.join(args.data_root, args.data_name[-6:], args.data_name + '_pose_stats.txt')
        # print("##############################",pose_stats_file)
        # print(np.loadtxt(pose_stats_file))
        pose_m, pose_s = np.loadtxt(pose_stats_file)
        
        # exit()
        
        torch.set_num_threads(2)
        
        try:
            main(args, start_epoch, end_epoch)
        except Exception as e:
            print(f"Error processing sequence {seq}: {str(e)}")
            continue
        finally:
            torch.cuda.empty_cache()
            gc.collect()
            
        print(f"\nFinished testing sequence: {seq}\n")
    
#CUDA_VISIBLE_DEVICES=0 python Eval_Global.py  --config config/QEOxford/sgloc.yaml