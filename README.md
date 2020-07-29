# Saliency Guided Self-attention Network for Weakly and Semi-supervised Semantic Segmentation (IEEE ACCESS)
This code is a implementation of the high-quality seed generation step in the paper [SGAN](https://arxiv.org/abs/1910.05475). The code is developed based on the Pytorch framework.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/saliency-guided-self-attention-network-for/weakly-supervised-semantic-segmentation-on)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on?p=saliency-guided-self-attention-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/saliency-guided-self-attention-network-for/weakly-supervised-semantic-segmentation-on-1)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on-1?p=saliency-guided-self-attention-network-for)

## Introduction
![Overview of SGAN](./images/graphical_abstract.png)
The proposed approach consists of three components: (1) a CNN backbone to learn deep feature representations; (2) a saliency guided self-attention module that propagates attentions from small discriminative parts to non-discriminative regions; (3) an image classification branch and a seed segmentation branch to supervise the training of the entire network.

## Installation
### 1. Prerequisites
Tested on Ubuntu16.04, CUDA9.0, python3.5, Pytorch 0.4.1, [CRF](https://github.com/kolesman/SEC), NVIDIA RTX 2080TI

### 2. Dataset
1. Download the prepared [VOC2012 dataset](https://drive.google.com/open?id=1PDTEuTnWJZNWogxYdqYGOlEZHK8dYET9), it contains all the files to reproduce the paper's results, including training images, pre-computed saliency maps and initial seeds.
2. modify the "dataset_root" variable in config file correspondingly.

### 3. models
1. Download [vgg_init.pth](https://drive.google.com/open?id=1lsr7Btwx_1bmc4T2QufLqjojuthOEYuM) model for initializing the SGAN network.
2. You can also download the trained SGAN model [model_iter_8000.pth](https://drive.google.com/open?id=193iExmZcxT7hkpVH4Pgo3KI2gkM_0MF8) for seed generation.

## Usage
### 1. train SGAN network
```bash
CUDA_VISIBLE_DEVICES=0,1 python tools/stage2/train_sgan.py  --cfg_file config/sgan_vgg16_321x321.yml
```

### 2. generate high-quality seeds
```bash
CUDA_VISIBLE_DEVICES=0 python tools/stage2/infer_cam.py --cfg_file config/sgan_vgg16_321x321.yml
```

### 3. (optionally) evaluate the seed quality
```bash
python tools/eval_mIoU.py --res_path [seed_path] --num_classes 22
```

## Seed Result
If you have finished all the steps above, you will get the seeds with precision=76.38 recall=57.26. The result is slightly different with that reported in our paper(precision=76.37, recall=57.35) since this repo is seperated from a larger project and the code may be slighted modified.

The pre-computed seeds can be downloaded [here](https://drive.google.com/open?id=10AU1YOsC8un99AeszM9UHbth3agV3IT5).

## Segmentation Result
If you want to reproduce our segmentation results, please refer [DSRG](https://github.com/speedinghzl/DSRG) for experiment setup, you may need to compile caffe, setup caffe python path and some other path variables. We provide the dsrg code, caffe with modified data layer in `dsrg` folder. In `experiment/anti_noise` folder, run
```
./run_anti_noise.sh [gpu_id]
```
Note that the our modified dsrg code is a little dirty and I don't have time to clean up. However, it's enough to reproduce the segmentation performance.

## License
SGAN is released under the MIT license

## Citing SGAN
if you find SGAN useful in your research, please consider citing:
```txt
@article{yao2020saliency,
  title={Saliency Guided Self-Attention Network for Weakly and Semi-Supervised Semantic Segmentation},
  author={Yao, Qi and Gong, Xiaojin},
  journal={IEEE Access},
  volume={8},
  pages={14413--14423},
  year={2020},
  publisher={IEEE}
}
```

if you also use the SNet to generate saliency maps, please consider citing:
```txt
@article{xiao2018deep,
  title={Deep salient object detection with dense connections and distraction diagnosis},
  author={Xiao, Huaxin and Feng, Jiashi and Wei, Yunchao and Zhang, Maojun and Yan, Shuicheng},
  journal={IEEE Transactions on Multimedia},
  volume={20},
  number={12},
  pages={3239--3251},
  year={2018},
  publisher={IEEE}
}
@inproceedings{huang2018dsrg,
    title={Weakly-Supervised Semantic Segmentation Network with Deep Seeded Region Growing},
    author={Huang, Zilong and Wang, Xinggang and Wang, Jiasi and Liu, Wenyu and Wang, Jingdong},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    pages={7014--7023},
    year={2018}
}
```
