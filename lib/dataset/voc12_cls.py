import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import scipy.ndimage as nd
import cv2
import PIL.Image
import os, random
import os.path as osp
import pickle
from scipy.spatial import distance
from scipy.stats import norm
import init_path
from lib.dataset import imutils
import matplotlib.pyplot as plt
plt.switch_backend("agg")

from ipdb import set_trace


class VOC12ClsDataset(Dataset):
    """voc2012 multi-label classification dataset"""
    def __init__(self, voc12_root, cue_file, max_iters=None, rescale_range=[448, 768, 448], crop_size=448,
                mirror=True, mean=(128, 128, 128), std=None, color_jitter=False):
        
        self.voc12_root = voc12_root
        self.label_dict = pickle.load(open(cue_file, "rb"))
        self.img_name_list = sorted(list(self.label_dict.keys()))
        if max_iters:
            self.img_name_list = self.img_name_list * int(np.ceil(float(max_iters) / len(self.img_name_list)))

        # data pre-processing
        self.rescale_range = rescale_range # [low_size, high_size, short_thresh]
        assert rescale_range[-1] >= crop_size, "short side after rescale must be larger than crop size"

        self.mirror = mirror
        self.crop_size = crop_size
        self.color_jitter = color_jitter
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.img_name_list)
    
    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        im_full_name = osp.join(self.voc12_root, "JPEGImages", name + ".jpg")
        im = cv2.imread(im_full_name)
        h, w = im.shape[:2]

        # color jitter
        if self.color_jitter:
            im = imutils.random_jitter(im)

        # rescale
        if len(self.rescale_range) == 1:
            new_size = self.rescale_range[0]
            im = nd.zoom(im, (new_size / h, new_size / w, 1.0), order=1)
        else:
            im = imutils.random_scale_with_size_keep_ratio(im, self.rescale_range[1], self.rescale_range[0], self.rescale_range[2])

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

        # random crop
        im = imutils.random_crop(im, None, self.crop_size, data_pad_value=(0, 0, 0), label_pad_value=(21,))

        # HWC->CHW
        im = im.transpose((2, 0, 1))

        # random flip
        if self.mirror:
            flip = np.random.choice(2) * 2 - 1
            if flip == -1:
                im = im[:, :, ::-1]

        label = np.zeros(shape=(20,), dtype=np.int)
        label_ind = self.label_dict[name]
        label[label_ind - 1] = 1

        return name, im.copy(), label.copy(), np.array([h, w])
    

if __name__ == "__main__":
    dataset_root = "/data1/yaoqi/Dataset/VOCdevkit/VOC2012/"
    cue_file = "im_tags.pkl"
    rescale_range = [512, 768, 448]
    short_thresh = 448
    save_path = "visualize/joint_vgg16_448x448"
    if not osp.exists(save_path):
        os.mkdir(save_path)

    train_dataset = VOC12ClsDataset(voc12_root=dataset_root,
                                    cue_file=cue_file,
                                    max_iters=100,
                                    rescale_range=rescale_range,
                                    crop_size=448,
                                    mirror=True,
                                    mean=(0, 0, 0),
                                    color_jitter=True)
    
    for k, pack in enumerate(train_dataset):
        name, im, label, size = pack
        f = plt.figure(facecolor="white")
        ax = f.add_subplot(111)
        im = np.transpose(im, axes=(1, 2, 0)).astype(np.uint8)
        ax.imshow(im[:, :, ::-1])
        ax.axis("off")
        ax.set_title(name)

        plt.tight_layout()
        plt.savefig(osp.join(save_path, name + ".png"))
        plt.close()

        if k == 10:
            break