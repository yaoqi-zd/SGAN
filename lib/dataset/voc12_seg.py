#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Author:  yaoqi (yaoqi_isee@zju.edu.cn)
# Created Date: 2019-11-24
# Modified By: yaoqi (yaoqi_isee@zju.edu.cn)
# Last Modified: 2019-12-15
# -----
# Copyright (c) 2019 Zhejiang University
"""
"""

import os, random
import os.path as osp
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import scipy.ndimage as nd
import init_path
from lib.dataset import imutils
import matplotlib.pyplot as plt
plt.switch_backend("agg")
from PIL import Image


class VOC12SegDataset(Dataset):
    """voc2012 segmentation dataset"""
    def __init__(self, im_root, label_root, train_id_list, max_iters=None, rescale_range=[448, 768, 448], crop_size=448, 
                 mean=(128, 128, 128), std=None, mirror=True, color_jitter=False, ignore_label=21, out_cue=False):
        self.im_root = im_root
        self.label_root = label_root
        self.name_list = [line.strip() for line in open(train_id_list, "r").readlines()]
        if max_iters:
            self.name_list = self.name_list * int(np.ceil(float(max_iters) / len(self.name_list)))

        # data pre-processing
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.mean = mean # in BGR order
        self.std = std
        self.mirror = mirror
        self.color_jitter = color_jitter
        self.ignore_label = ignore_label
        self.out_cue = out_cue


    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        im_full_name = osp.join(self.im_root, self.name_list[idx] + ".jpg")
        label_full_name = osp.join(self.label_root, self.name_list[idx] + ".png")
        img = cv2.imread(im_full_name, cv2.IMREAD_COLOR)
        label = cv2.imread(label_full_name, cv2.IMREAD_GRAYSCALE)

        # color jitter
        if self.color_jitter:
            img = imutils.random_jitter(img)

        # rescale
        h, w = img.shape[:2]
        if len(self.rescale_range) == 1:
            new_size = self.rescale_range[0]
            img = nd.zoom(img, (new_size / h, new_size / w, 1.0), order=1)
            label = nd.zoom(label, (new_size / h, new_size / w), order=0)
        else:
            img, label = imutils.random_scale_with_size_keep_ratio(img, self.rescale_range[1], \
                self.rescale_range[0], self.rescale_range[2], label)

        # # subtract mean value
        # img = np.asarray(img, np.float32)
        # img -= self.mean
        # normalize
        if self.std is not None:
            img = img.astype(np.float)
            img = img[:, :, ::-1] # change to RGB order
            img /= 255.0
            img -= self.mean
            img /= self.std
        else:
            img = img.astype(np.float)
            img -= self.mean

        # random crop
        img, label = imutils.random_crop(img, label, self.crop_size, (0, 0, 0), (self.ignore_label,))

        # HWC->CHW
        img = img.transpose((2, 0, 1))

        # random flip
        if self.mirror:
            flip = np.random.choice(2) * 2 - 1
            img = img[:, :, ::flip]
            label = label[:, ::flip]

        # preprocess initial seed
        if self.out_cue:
            feat_size = round(self.crop_size / 8 + 0.5)
            seed_reduced = nd.zoom(label, (feat_size / label.shape[0], feat_size / label.shape[1]), order=0)
            
            non_ignore_ind = np.where(seed_reduced != 21)
            cues = np.zeros(shape=(21, feat_size, feat_size), dtype=np.float)
            cues[seed_reduced[non_ignore_ind], non_ignore_ind[0], non_ignore_ind[1]] = 1.0

            return self.name_list[idx], img.copy(), cues.copy()
        else:
            return self.name_list[idx], img.copy(), label.copy()


class VOC12ValSegDataset(Dataset):
    def __init__(self, voc12_root, val_list, new_size=321, mean=(128, 128, 128), std=None):
        self.voc12_root = voc12_root
        self.img_name_list = [line.strip() for line in open(val_list, "r").readlines()]
        self.new_size = new_size
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        im_full_name = osp.join(self.voc12_root, "JPEGImages", name + ".jpg")
        im = cv2.imread(im_full_name)
        h, w = im.shape[:2]

        gt_full_name = osp.join(self.voc12_root, "SegmentationClass", name + ".png")
        # gt = cv2.imread(gt_full_name, cv2.IMREAD_GRAYSCALE)
        gt = np.array(Image.open(gt_full_name))
        
        # resize
        im = nd.zoom(im, (self.new_size / h, self.new_size / w, 1.0), order=1)
        gt = nd.zoom(gt, (self.new_size / h, self.new_size / w), order=0)

        # normalize
        if self.std is not None:
            im = im.astype(np.float)
            im = im[:, :, ::-1] # change to RGB order
            im /= 255.0
            im -= self.mean
            im /= self.std
        else:
            im = im.astype(np.float)
            im -= self.mean

        # HWC->CHW
        im = im.transpose((2, 0, 1))

        return name, im.copy(), gt.copy()

class VOC12TestSegDataset(Dataset):
    """voc2012 test dataset"""
    def __init__(self, voc12_root, val_list, crop_size=513, mean=(128, 128, 128), ignore_label=255):
        self.voc12_root = voc12_root
        self.img_name_list = [line.strip() for line in open(val_list, "r").readlines()]
        self.crop_size = crop_size
        self.mean = np.asarray(mean, dtype=np.float32).reshape((1, 1, 3))  # in BGR order
        self.ignore_label = ignore_label

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        im_full_name = osp.join(self.voc12_root, "JPEGImages", name + ".jpg")
        im = cv2.imread(im_full_name)
        h, w = im.shape[:2]

        # subtract mean
        im = np.asarray(im, np.float32)
        im -= self.mean

        # crop
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        assert pad_h > 0 or pad_w > 0

        im = cv2.copyMakeBorder(im, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # im = nd.zoom(im, (321.0 / h, 321.0 / w, 1), order=1)

        # HWC->CHW
        im = im.transpose((2, 0, 1))

        return name, im.copy(), np.array([h, w])



if __name__ == "__main__":
    root = "/data/yaoqi/Dataset/VOCdevkit/VOC2012"
    im_root = osp.join(root, "JPEGImages")
    label_root = osp.join(root, "SegmentationClassAug")
    train_id_list = "list/train_id.txt"
    val_id_list = "list/val_id.txt"
    max_iter = 8000
    rescale_range = [448, 768, 448]
    mean = (0, 0, 0)

    save_path = "visualize/viz_seg_loader"
    if not osp.exists(save_path):
        os.mkdir(save_path)

    # dataset = VOC12SegDataset(im_root=im_root,
    #                           label_root=label_root,
    #                           train_id_list=train_id_list,
    #                           max_iters=max_iter,
    #                           rescale_range=rescale_range,
    #                           crop_size=448,
    #                           mean=mean,
    #                           color_jitter=True)
    
    dataset = VOC12ValSegDataset(voc12_root=root, val_list=val_id_list, new_size=321, mean=mean)

    for k in range(len(dataset)):
        name, im, label = dataset[k]
        im = im.astype(np.uint8).transpose((1, 2, 0))

        f = plt.figure(facecolor="white")    
        ax = f.add_subplot(1, 2, 1)
        ax.imshow(im[:, :, ::-1])
        ax.axis("off")

        ax = f.add_subplot(1, 2, 2)
        ax.matshow(label)
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(osp.join(save_path, name + ".png"))
        plt.close()

        if k == 10:
            break
        

    