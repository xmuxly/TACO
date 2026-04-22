<p align="center">
  <h2 align="center">TACO: Task-Aware Contrastive Learning for Joint LiDAR Localization and 3D Object Detection</h2>
  <h3 align="center">CVPR 2026</h3>

<div align="center">
<a alighn="center" <img src='https://img.shields.io/badge/CVF-Paper-blue' alt='Paper PDF'></a>
</p>
</div>

## ⚙️ Environment
```bash
conda create -n spconv2 python=3.8
conda activate spconv2
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy==1.19.5 protobuf==3.19.4 scikit-image==0.19.2 waymo-open-dataset-tf-2-5-0 nuscenes-devkit==1.0.5 spconv-cu111 numba scipy pyyaml easydict fire tqdm shapely matplotlib opencv-python addict pyquaternion awscli open3d pandas future pybind11 tensorboardX tensorboard Cython prefetch-generator
```
Environment we tested:
```
Ubuntu 18.04
Python 3.8.13
PyTorch 1.8.1
Numba 0.53.1
Spconv 2.1.22 # pip install spconv-cu111
NVIDIA CUDA 11.1
4x 3090 GPUs
```

## Prepare Dataset

TACO currently supports the following datasets:
- [Oxford Radar RobotCar](https://oxford-robotics-institute.github.io/radar-robotcar-dataset/datasets)
- [NuScenes]
- [KITTI-360]
### Dataset Structure
Organize the dataset directories as follows:

- (QE)Oxford
```
data_root
├── 2019-01-11-14-02-26-radar-oxford-10k
│   ├── velodyne_left
│   │   ├── xxx.bin
│   │   ├── xxx.bin
│   │   ├── …
│   ├── id_label
│   │   ├── xxx.txt
│   │   ├── xxx.txt
│   │   ├── …
│   ├── label_mot
│   │   ├── xxx.txt
│   │   ├── xxx.txt
│   │   ├── …
│   ├── label_m
│   │   ├── xxx.txt
│   │   ├── xxx.txt
│   │   ├── …
│   ├── velodyne_left_calibrateFalse.h5
│   ├── velodyne_left_False.h5
│   ├── rot_tr.bin
│   ├── tr.bin
│   ├── tr_add_mean.bin
├── …
├── (QE)Oxford_pose_stats.txt
├── train_split.txt
├── valid_split.txt
```


## 💃 Run
### train
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.34 --master_port 29503 train_ddp.py
```
## Evaluation
```
python Eval_Loc.py 
python Eval_Det.py
```
%## 🌟 Visualization

%## 🤗 Model zoo

%## 🙏 Acknowledgements

 We appreciate the code of [SGLoc](https://github.com/liw95/SGLoc/tree/main), [LiSA]()
## 🎓 Citation
