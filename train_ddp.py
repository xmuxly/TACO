import sys
sys.path.append('/home/ssd1/code/anchor_SG/OpenPCDet')  # 指向pcdet所在的父目录

# pylint: disable=no-member
import argparse
import os
import sys
import numpy as np
import random
import torch
import matplotlib
import torch.nn as nn
import math
import time  # 添加time模块
from collections import defaultdict  # 添加defaultdict
from torch.cuda.amp import GradScaler, autocast

matplotlib.use('Agg')
# Oxford
from data.oxford import Oxford
# QEOxford
from data.qeoxford import QEOxford
# NCLT
from data.nclt import NCLT
from util import config
from model.lisa import LiSA, AlignedContrastiveLoss, DetContrastiveLoss, LocContrastiveLoss
from model.loss import L1_CriterionCoordinate, CriterionCoordinate, DDPM_CriterionCoordinate
from data.dataset import sgloc_data, collate_fn_default
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tqdm import tqdm

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.34 --master_port 29503 train_ddp.py >> log.txt 2>&1

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Localization')
    # parser.add_argument('--config', type=str, default='config/nuScenes/nuscenes.yaml',
    #                     help='config file')    
    parser.add_argument('--config', type=str, default='config/QEOxford/TACO.yaml',
                        help='config file')
    parser.add_argument('--local_rank', type=int, default=0,
                        help="If using ddp to train")
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg.update(vars(args))
    return cfg

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    # print(out_str)


def main(args):
    global TOTAL_ITERATIONS
    TOTAL_ITERATIONS = 0

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    device = torch.device(f'cuda:{args.local_rank}')

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.enabled = True
        cudnn.deterministic = True

    for (k, v) in args.items():
        print('%s: %s' % (str(k), str(v)))

    if args.data_name == 'Oxford':
        train_pt_dataset = Oxford(args.data_root,
                                    split='train')
    elif args.data_name == 'QEOxford':
        train_pt_dataset = QEOxford(args.data_root,
                                  split='train')
    elif args.data_name == 'NCLT':
        train_pt_dataset = NCLT(args.data_root,
                                  split='train')
    else:
        print('Only supprot Oxford, QEOxford and NCLT')

    dataset = sgloc_data(train_pt_dataset, args)
    tran_sampler = DistributedSampler(dataset)

    train_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               worker_init_fn=my_worker_init_fn,
                                               pin_memory=True,
                                            #    prefetch_factor=2,  # 预取2个batch的数据
                                            #    persistent_workers=True,  # 保持worker进程存活                                              
                                               collate_fn=collate_fn_default,
                                               sampler=tran_sampler,
                                               drop_last=True,  # 丢弃不完整batch
                                               )

    # pose_stats_file = os.path.join(args.data_root, args.data_name + '_pose_stats.txt')
    # pose_m, pose_s = np.loadtxt(pose_stats_file)

    train_writer = SummaryWriter(os.path.join(args.save_path, 'train'))

    # model = LiSA_ddpm(args)
    model = LiSA(args)

    model = model.to(device=args.local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      broadcast_buffers=False,
                                                      find_unused_parameters=False)
    loss = L1_CriterionCoordinate()
    alignedContrastiveLoss = AlignedContrastiveLoss()
    detContrastiveLoss = DetContrastiveLoss()
    locContrastiveLoss = LocContrastiveLoss()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.base_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_step, gamma=0.95)

    scaler = GradScaler()

    LOG_FOUT.write("\n")
    LOG_FOUT.flush()
    
    ################
    # 设置起始epoch
    start_epoch = 0
    # 只在主进程中打印信息
    is_main_process = args.local_rank == 0
    ################
    # 判断是否需要加载预训练模型
    if args.resume_training:
        if os.path.exists(args.pretrained_path):
            print(f"Loading pretrained weights from {args.pretrained_path}")
            checkpoint = torch.load(args.pretrained_path, map_location='cpu')
            # 1. 加载模型权重
            saved_state_dict = checkpoint['state_dict']
            # 处理权重的键名，添加"module."前缀
            for k in list(saved_state_dict.keys()):
                saved_state_dict[f"module.{k}"] = saved_state_dict.pop(k)
            # 加载权重
            model.load_state_dict(saved_state_dict, strict=False)
            print("Successfully loaded pretrained weights")
                
            # 2. 加载优化器状态
            optimizer.load_state_dict(checkpoint['optimizer'])
            # 3. 加载学习率调度器状态
            scheduler.load_state_dict(checkpoint['scheduler'])
            # 4. 恢复训练迭代次数
            TOTAL_ITERATIONS = checkpoint['iter']
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")
             # 集中打印信息
            if is_main_process:
                print(f"Loading checkpoint from {args.pretrained_path}")
                print("Successfully loaded pretrained weights")
                print(f"Resuming training from epoch {start_epoch}")
        else:
            if is_main_process:
                print(f"No pretrained weights found at {args.pretrained_path}")
                print("Training from scratch")
    else:
        if is_main_process:
            print("Training from scratch")
        # 等待所有进程
    torch.distributed.barrier()
    ###############
    
    
    for epoch in range(start_epoch, args.epochs):
        log_string('**** EPOCH %03d ****' % epoch)
        sys.stdout.flush()
        train_one_epoch(model, train_loader, tran_sampler, scheduler, epoch, train_writer, loss, args, device, scaler, alignedContrastiveLoss, detContrastiveLoss, locContrastiveLoss)


def train_one_epoch(model, train_loader, tran_sampler, scheduler, epoch, train_writer, loss, args, device, scaler, alignedContrastiveLoss, detContrastiveLoss, locContrastiveLoss):
    global TOTAL_ITERATIONS
    model.train()
    tran_sampler.set_epoch(epoch)
    tqdm_loader = tqdm(train_loader, total=len(train_loader))
    timers = defaultdict(float)
    num_iters = 0

    for _, input_dict in enumerate(tqdm_loader):
        num_iters += 1
        TOTAL_ITERATIONS += 1
        
        # 记录数据加载开始时间
        data_start = time.time()
        
        input_dict['points'] = input_dict['points'].to(device, dtype=torch.float32)
        # print(input_dict['points'])
        # print(input_dict['points'].shape)
        input_dict['batch_idx'] = input_dict['batch_idx'].to(device, dtype=torch.int8)
        input_dict['labels'] = input_dict['labels'].to(device, dtype=torch.float32)
        input_dict['gt_boxes'] = torch.from_numpy(input_dict['gt_boxes']).to(device, dtype=torch.float32)
        # print(input_dict['gt_boxes'])
        # print(input_dict['gt_boxes'].shape)
        
        
        # 记录数据加载结束时间
        data_time = time.time() - data_start
        timers['data_load'] += data_time
        
        scheduler.optimizer.zero_grad()
        
        # 记录模型前向传播开始时间
        forward_start = time.time()
        
        with autocast():
            pred_shift, data_dict = model(input_dict)
            labels = data_dict['labels']
            sup_point = labels[:, :3]
            gt_sup_point = labels[:, 3:6]
            seg_feature = labels[:, 6:]
            pred_point = sup_point + pred_shift
            loc_loss = loss(pred_point, gt_sup_point)
            det_loss, _ = model.module.AnchorHeadSingle.get_loss()

            # 定位-检测对比学习损失
            loc_proj, det_proj = data_dict['contrast_data']
            contrast_loss = alignedContrastiveLoss(
                data_dict = data_dict
            )
            
            # 动态-静态检测目标对比学习损失
            contrast_det_loss = detContrastiveLoss(
                data_dict = data_dict
            )
            
            # 静态目标对应定位特征-地理特征对比学习损失
            contrast_loc_loss = locContrastiveLoss(
                data_dict = data_dict
            )
            
            train_loss = loc_loss + det_loss + contrast_loss + contrast_det_loss + contrast_loc_loss
            
        print("contrast_loss: ", contrast_loss)
        print("contrast_det_loss: ", contrast_det_loss)
        print("contrast_loc_loss: ", contrast_loc_loss)
        
        forward_time = time.time() - forward_start
        timers['forward'] += forward_time

        # 反向传播（混合精度）
        backward_start = time.time()
        scaler.scale(train_loss).backward()

        # # 梯度裁剪
        # scaler.unscale_(scheduler.optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(scheduler.optimizer)
        scaler.update()
        scheduler.step()
        
        # 记录反向传播结束时间
        backward_time = time.time() - backward_start
        timers['backward'] += backward_time
        
        # 更新进度条信息
        tqdm_loader.set_description(
            f'Epoch {epoch} | Data: {data_time:.3f}s | Forward: {forward_time:.3f}s | Backward: {backward_time:.3f}s'
        )
        
        log_string('Loss: %f' % train_loss)
        log_string('contrast_loss: %f' % contrast_loss)
        log_string('contrast_det_loss: %f' % contrast_det_loss)
        log_string('contrast_loc_loss: %f' % contrast_loc_loss)
        train_writer.add_scalar('Loss', train_loss.cpu().item(), TOTAL_ITERATIONS)
        
        # 每100次迭代输出一次平均时间统计
        if num_iters % 100 == 0 and torch.distributed.get_rank() == 0:
            avg_data_time = timers['data_load'] / num_iters
            avg_forward_time = timers['forward'] / num_iters
            avg_backward_time = timers['backward'] / num_iters
            log_string(f'Average timing stats after {num_iters} iterations:')
            log_string(f'Data loading: {avg_data_time:.3f}s/iter')
            log_string(f'Forward pass: {avg_forward_time:.3f}s/iter')
            log_string(f'Backward pass: {avg_backward_time:.3f}s/iter')
            log_string(f'Total: {(avg_data_time + avg_forward_time + avg_backward_time):.3f}s/iter')

    if epoch % 1 == 0 and torch.distributed.get_rank() == 0:
        model_to_save = model.module
        torch.save({
            'epoch': epoch,
            'iter': TOTAL_ITERATIONS,
            'state_dict': model_to_save.state_dict(),
            'optimizer': scheduler.optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        },
            args.save_path + '/checkpoint_epoch{}.tar'.format(epoch))
        print("Model Saved As " + 'checkpoint_epoch{}.tar'.format(epoch))
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import gc

    gc.collect()
    args = get_parser()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    LOG_FOUT = open(os.path.join(args.save_path, 'log_contract_t.txt'), 'a' if args.resume_training else 'w')
    # 如果是继续训练，添加分隔符
    if args.resume_training:
        LOG_FOUT.write('\n' + '='*50 + '\n')
        LOG_FOUT.write(f'Resume training from checkpoint: {args.pretrained_path}\n')
        LOG_FOUT.write('='*50 + '\n\n')
    else:
        LOG_FOUT.write(str(args) + '\n')
    # 占用2个核心
    torch.set_num_threads(4)
    main(args)