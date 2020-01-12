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

from ipdb import set_trace

class VOC12ClsSalDataset(Dataset):
    """voc2012 multi-label classification dataset with saliency as additional information"""
    def __init__(self, voc12_root, cue_file, max_iters=None, new_size=321, mirror=True, mean=(128, 128, 128), std=None,
                sal_subdir="sal_sdnet", seed_subdir="base_30_bg", sal_thresh=0.06):
        self.voc12_root = voc12_root
        self.label_dict = pickle.load(open(cue_file, "rb"))
        self.sal_subdir = sal_subdir
        self.sal_thresh = sal_thresh
        self.seed_subdir = seed_subdir
        self.img_name_list = sorted(list(self.label_dict.keys()))
        if max_iters:
            self.img_name_list = self.img_name_list * int(np.ceil(float(max_iters) / len(self.img_name_list)))

        # data pre-processing
        self.mirror = mirror
        self.new_size = new_size
        self.mean = mean  # in BGR order
        self.std = std

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        im_full_name = osp.join(self.voc12_root, "JPEGImages", name + ".jpg")
        sal_full_name = osp.join(self.voc12_root, "sal", self.sal_subdir, name + ".png")
        seed_full_name = osp.join(self.voc12_root, "ini_seed", self.seed_subdir, name + ".png")
        im = cv2.imread(im_full_name)
        sal = cv2.imread(sal_full_name, cv2.IMREAD_GRAYSCALE)
        seed = cv2.imread(seed_full_name, cv2.IMREAD_GRAYSCALE)
        h, w = im.shape[:2]

        # resize
        im = nd.zoom(im, (self.new_size / h, self.new_size / w, 1), order=1)
        sal = nd.zoom(sal, (self.new_size / h, self.new_size / w), order=1)
        seed = nd.zoom(seed, (self.new_size / h, self.new_size / w), order=0)

        # subtract mean
        im = np.asarray(im, np.float32)
        if self.std is not None:
            im = im[:, :, ::-1] # change to RGB order
            im /= 255.0
            im -= self.mean
            im /= self.std
        else:
            im -= self.mean

        # for sal from sdnet
        # sal = np.asarray(sal, np.float32)
        # sal /= 255
        # sal_bi = np.ones(sal.shape)
        # sal_bi[sal > self.sal_thresh] = 1
        # sal_bi[sal <= self.sal_thresh] = 0
        # sal_bi = sal_bi.astype(np.uint8)

        # HWC->CHW
        im = im.transpose((2, 0, 1))

        # random flip
        if self.mirror:
            flip = np.random.choice(2) * 2 - 1
            im = im[:, :, ::flip]
            sal = sal[:, ::flip]
            seed = seed[:, ::flip]

        label = np.zeros(shape=(20,), dtype=np.int)
        label_ind = self.label_dict[name]
        label[label_ind - 1] = 1

        # for sal to compute similarity map
        feat_size = round(self.new_size / 8 + 0.5)
        sal_reduced = nd.zoom(sal, (feat_size / self.new_size, feat_size / self.new_size), order=1).astype(np.float32)
        sal_reduced /= 255.0
        sal_reduced = sal_reduced > self.sal_thresh
        sal_reduced = sal_reduced.astype(np.uint8)

        sal_reduced[:1, :] = 0
        sal_reduced[:, :1] = 0
        sal_reduced[-1:, :] = 0
        sal_reduced[:, -1:] = 0

        sal_reshape = np.tile(np.reshape(sal_reduced, (feat_size * feat_size, -1)), (1, feat_size * feat_size))
        sal_reshape_1 = np.transpose(sal_reshape)

        # divide into fg sim and bg sim
        fg_sim = sal_reshape * sal_reshape_1
        bg_sim = (1 - sal_reshape) * (1 - sal_reshape_1)
        fg_sim = fg_sim.astype(np.float32)
        bg_sim = bg_sim.astype(np.float32)
        bg_sim = bg_sim / (np.sum(bg_sim, axis=-1) + 1e-5)

        # preprocess initial seed
        seed_reduced = nd.zoom(seed, (feat_size / self.new_size, feat_size / self.new_size), order=0)
        zero_ind = np.where(seed_reduced == 0)
        seed_reduced[zero_ind] = 21
        seed_reduced = seed_reduced - 1

        return name, im.copy(), label.copy(), fg_sim.copy(), bg_sim.copy(), seed_reduced.copy(), np.array([h, w])


class VOC12TestSalDataset(Dataset):
    """voc2012 test dataset"""

    def __init__(self, voc12_root, cue_file, new_size=321, mean=(128, 128, 128)):
        self.voc12_root = voc12_root
        self.label_dict = pickle.load(open(cue_file, "rb"))
        self.img_name_list = sorted(list(self.label_dict.keys()))
        self.new_size = new_size
        self.mean = np.asarray(mean, dtype=np.float32).reshape((1, 1, 3))  # in BGR order

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        im_full_name = osp.join(self.voc12_root, "JPEGImages", name + ".jpg")
        sal_full_name = osp.join(self.voc12_root, "sal_sdnet", name + ".png")
        im = cv2.imread(im_full_name)
        sal = cv2.imread(sal_full_name, cv2.IMREAD_GRAYSCALE)
        h, w = im.shape[:2]

        # resize
        im = nd.zoom(im, (self.new_size / h, self.new_size / w, 1), order=1)
        sal = nd.zoom(sal, (self.new_size / h, self.new_size / w), order=1)

        # subtract mean
        im = np.asarray(im, np.float32)
        im -= self.mean

        # HWC->CHW
        im = im.transpose((2, 0, 1))

        label = np.zeros(shape=(20,), dtype=np.int)
        label_ind = self.label_dict[name]
        label[label_ind - 1] = 1

        # for sal to compute similarity map
        sal = np.asarray(sal, np.float32)
        sal /= 255
        sal_bi = np.ones(sal.shape) * 21
        sal_bi[sal > 0.06] = 1
        sal_bi[sal <= 0.06] = 0
        sal_bi = sal_bi.astype(np.uint8)

        feat_size = round(self.new_size / 4 + 0.5)
        sal_reduced = nd.zoom(sal_bi, (feat_size / self.new_size, feat_size / self.new_size), order=0)
        sal_reshape = np.tile(np.reshape(sal_reduced, (feat_size * feat_size, -1)), (1, feat_size * feat_size))
        sal_reshape_1 = np.transpose(sal_reshape)

        # sal_reshape = np.tile(np.reshape(sal_bi, (self.new_size * self.new_size, -1)), (1, self.new_size * self.new_size))
        # sal_reshape_1 = np.transpose(sal_reshape)

        # divide into fg sim and bg sim
        fg_sim = sal_reshape * sal_reshape_1
        bg_sim = (1 - sal_reshape) * (1 - sal_reshape_1)
        fg_sim = fg_sim.astype(np.float32)
        bg_sim = bg_sim.astype(np.float32)
        bg_sim = bg_sim / (np.sum(bg_sim, axis=-1) + 1e-5)

        return name, im.copy(), label.copy(), fg_sim.copy(), bg_sim.copy(), np.array([h, w])


# class VOC12ClsSeedDataset(Dataset):
#     """voc2012 multi-label classification with seed as additional information"""
#     def __init__(self, voc12_root, cue_file, seed_subdir=None, max_iters=None, resize=353,
#                 crop_size=321, mstrain=[], mirror=True, mean=(128, 128, 128), color_jitter=False,
#                 out_cue=False, ignore_label=21, use_att_mask=False):
#         self.voc12_root = voc12_root
#         self.seed_subdir = seed_subdir
#         self.label_dict = pickle.load(open(cue_file, "rb"))
#         self.img_name_list = sorted(list(self.label_dict.keys()))
#         if max_iters:
#             self.img_name_list = self.img_name_list * int(np.ceil(float(max_iters) / len(self.img_name_list)))

#         # data pre-processing
#         self.mirror = mirror
#         self.crop_size = crop_size
#         self.new_size = resize
#         self.mstrain = mstrain
#         self.color_jitter = color_jitter
#         self.mean = np.asarray(mean, dtype=np.float32).reshape((1, 1, 3))  # in BGR order

#         self.out_cue = out_cue # produce localization cues format (c * h * w) or segemntation mask format (h * w)
#         self.ignore_label = ignore_label
#         self.use_att_mask = use_att_mask # whether to output fore-ignore-background mask (1, hw) for balanced self-attention

#     def __len__(self):
#         return len(self.img_name_list)

#     def __getitem__(self, idx):
#         name = self.img_name_list[idx]
#         im_full_name = osp.join(self.voc12_root, "JPEGImages", name + ".jpg")
#         seed_full_name = osp.join(self.voc12_root, self.seed_subdir, name + ".png")
#         assert osp.exists(seed_full_name), "{} not exists".format(seed_full_name)
#         im = cv2.imread(im_full_name)
#         seed = cv2.imread(seed_full_name, cv2.IMREAD_GRAYSCALE)
#         h, w = im.shape[:2]

#         # resize
#         im = nd.zoom(im, (self.new_size / h, self.new_size / w, 1), order=1)
#         seed = nd.zoom(seed, (self.new_size / h, self.new_size / w), order=0)        

#         # color jitter
#         if self.color_jitter:
#             im = imutils.random_jitter(im)

#         # subtract mean
#         im = np.asarray(im, np.float32)
#         im -= self.mean

#         # crop
#         im, seed = imutils.random_crop(im, seed, self.crop_size, data_pad_value=(0, 0, 0), label_pad_value=(self.ignore_label,))
        
#         # HWC->CHW
#         im = im.transpose((2, 0, 1))

#         # random flip
#         if self.mirror:
#             flip = np.random.choice(2) * 2 - 1
#             if flip == -1:
#                 im = im[:, :, ::-1]
#                 seed = seed[:, ::-1]

#         label = np.zeros(shape=(20,), dtype=np.int)
#         label_ind = self.label_dict[name]
#         label[label_ind - 1] = 1

#         # preprocess initial seed
#         feat_size = round(self.crop_size / 8 + 0.5)
#         seed_reduced = nd.zoom(seed, (feat_size / seed.shape[0], feat_size / seed.shape[1]), order=0)

#         if self.out_cue:
#             non_ignore_ind = np.where(seed_reduced != 21)
#             cues = np.zeros(shape=(21, feat_size, feat_size), dtype=np.float)
#             cues[seed_reduced[non_ignore_ind], non_ignore_ind[0], non_ignore_ind[1]] = 1.0

#             if self.use_att_mask:
#                 att_mask = imutils.generate_att_mask_from_seed(seed_reduced)
#                 att_mask = np.reshape(att_mask, newshape=(1, -1))
#                 return name, im.copy(), label.copy(), cues.copy(), np.array([h, w]), att_mask.copy()
#             return name, im.copy(), label.copy(), cues.copy(), np.array([h, w])
#         else:
#             if self.use_att_mask:
#                 att_mask = imutils.generate_att_mask_from_seed(seed_reduced)
#                 att_mask = np.reshape(att_mask, newshape=(1, -1))
#                 return name, im.copy(), label.copy(), seed_reduced.copy(), np.array([h, w]), att_mask.copy()
#             return name, im.copy(), label.copy(), seed_reduced.copy(), np.array([h, w])

# class VOC12TestDataset(Dataset):
#     """voc2012 multi-label classification dataset"""
#     def __init__(self, voc12_root, cue_file, new_size=321, mean=(128, 128, 128)):
#         self.voc12_root = voc12_root
#         self.label_dict = pickle.load(open(cue_file, "rb"))
#         self.img_name_list = sorted(list(self.label_dict.keys()))
#         self.new_size = new_size
#         self.mean = np.asarray(mean, dtype=np.float32).reshape((1, 1, 3))  # in BGR order

#     def __len__(self):
#         return len(self.img_name_list)

#     def __getitem__(self, idx):
#         name = self.img_name_list[idx]
#         im_full_name = osp.join(self.voc12_root, "JPEGImages", name + ".jpg")
#         im = cv2.imread(im_full_name)
#         h, w = im.shape[:2]

#         # resize
#         im = nd.zoom(im, (self.new_size / h, self.new_size / w, 1), order=1)

#         # subtract mean
#         im = np.asarray(im, np.float32)
#         im -= self.mean

#         # HWC->CHW
#         im = im.transpose((2, 0, 1))

#         label = np.zeros(shape=(20,), dtype=np.int)
#         label_ind = self.label_dict[name]
#         label[label_ind - 1] = 1

#         return name, im.copy(), label.copy(), np.array([h, w])

# def denormalizing(ori_images, mean=(0, 0, 0), std=None):
#     """
#     denormalizing tensor images with mean and std
#     Args:
#         images: tensor, N*C*H*W
#         mean: tuple
#         std: tuple
#     """

#     images = torch.tensor(ori_images, requires_grad=False)

#     if std is not None:
#         """pytorch style normalization"""
#         mean = torch.tensor(mean).view((1, 3, 1, 1))
#         std = torch.tensor(std).view((1, 3, 1, 1))

#         images *= std
#         images += mean
#         images *= 255

#         images = torch.flip(images, dims=(1,))
#     else:
#         """caffe style normalization"""
#         mean = torch.tensor(mean).view((1, 3, 1, 1))
#         images += mean

#     return images

if __name__ == '__main__':
    import imutils
    import matplotlib.pyplot as plt
    plt.switch_backend("agg")
    from ipdb import set_trace
    # from tools import imutils

    voc12_root = "/data1/yaoqi/Dataset/VOCdevkit/VOC2012/"
    cue_file = "im_tags.pkl"
    batch_size = 2
    max_iter = 8000
    resize = 353
    crop_size = 321
    color_jitter = True
    save_path = "visualize/viz_dataloader"
    if not osp.exists(save_path):
        os.mkdir(save_path)

    classes = ['background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor']

    # test seed dataset
    train_dataset = VOC12ClsSeedDataset(voc12_root=voc12_root,
                                        cue_file=cue_file,
                                        max_iters=max_iter * batch_size, 
                                        resize=resize,
                                        crop_size=crop_size,
                                        color_jitter=color_jitter,
                                        seed_subdir="ini_seed/ssenet_tb",
                                        mean=(0, 0, 0),
                                        out_cue=False,
                                        use_att_mask=True)
    
    for k, data in enumerate(train_dataset):
        name, im, label, seed, size, att_mask = data
        im = np.transpose(im, (1, 2, 0)).astype(np.uint8)
        
        f = plt.figure(facecolor="white")
        ax = f.add_subplot(1, 3, 1)
        ax.imshow(im[:, :, ::-1])
        ax.axis("off")

        ax = f.add_subplot(1, 3, 2)
        ax.matshow(seed)
        ax.axis("off")

        ax = f.add_subplot(1, 3, 3)
        att_mask = np.reshape(att_mask, newshape=seed.shape)
        ax.imshow(att_mask, cmap=plt.cm.Greys_r)
        ax.axis("off")

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.01, hspace=0.15)
        plt.savefig(os.path.join(save_path, name + ".png"))
        plt.close()

        if k == 40:
            break

    # for k, data in enumerate(train_dataset):
    #     name, im, label, cue, size = data[0], data[1], data[2], data[3], data[4]
    #     im = np.transpose(im, (1, 2, 0)).astype(np.uint8)

    #     label_ind = np.where(label == 1)[0]
    #     cols = len(label_ind)

    #     f = plt.figure(facecolor="white")
    #     ax = f.add_subplot(1, cols + 1, 1)
    #     ax.imshow(im[:, :, ::-1])
    #     ax.axis("off")

    #     for m in range(cols):
    #         ax = f.add_subplot(1, cols + 1, m + 2)
    #         mask = cue[label_ind[m] + 1, :, :]
    #         mask = np.array(mask, dtype=np.int)
    #         ax.matshow(mask)
    #         ax.axis("off")
    #         ax.set_title(classes[label_ind[m] + 1])

    #     # plt.show()
    #     plt.tight_layout()
    #     plt.subplots_adjust(wspace=0.01, hspace=0.15)
    #     plt.savefig(os.path.join(save_path, name + ".png"))
    #     plt.close()

    #     if k == 10:
    #         break