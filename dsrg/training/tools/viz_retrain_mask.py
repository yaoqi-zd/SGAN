import os, argparse
import os.path as osp
import numpy as np
import scipy.ndimage as nd
import pylab, png, cPickle
from shutil import copyfile
from PIL import Image

from ipdb import set_trace
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import matplotlib.colors as mpl_colors

palette = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.5, 0.0),
           (0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5), (0.5, 0.5, 0.5),
           (0.25, 0.0, 0.0), (0.75, 0.0, 0.0), (0.25, 0.5, 0.0), (0.75, 0.5, 0.0),
           (0.25, 0.0, 0.5), (0.75, 0.0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5),
           (0.0, 0.25, 0.0), (0.5, 0.25, 0.0), (0.0, 0.75, 0.0), (0.5, 0.75, 0.0),
           (0.0, 0.25, 0.5), (0.75, 0.75, 0.75)]
my_cmap = mpl_colors.LinearSegmentedColormap.from_list('Custom cmap', palette, 22)

import cv2

im_path = "/home/yaoqi/Dataset/VOC2012/JPEGImages/"
gt_path = "/home/yaoqi/Dataset/VOC2012/SegmentationClassAug_color/"
init_seed_path = "/data1/yaoqi/home_segmentation/segmentation/weakly/DSRG/training/localization_cues/seed_0328_7440_5996/"
pred_path = "/data/yaoqi/segmentation/dsrg/result/seg_res_trainaug_6399/"
pred_tag_path = "experiment/anti-noise/result/seg_res_trainaug_6399_nocrf/"
crf_pred_path = "experiment/anti-noise/result/seg_res_trainaug_6399_seed_mIoU_6551_color"
save_path = "visualize/viz_retrain_mask"
if not osp.exists(save_path):
    os.makedirs(save_path)

im_score_file = "visualize/im_score_6396.pkl"

im_score = cPickle.load(open(im_score_file, "rb"))

im_names = os.listdir(gt_path)

for k, name in enumerate(im_names):
    im = cv2.imread(osp.join(im_path, name.replace("png", "jpg")))
    gt = cv2.imread(osp.join(gt_path, name))
    init_seed = np.array(Image.open(osp.join(init_seed_path, name)))
    pred_tag = cv2.imread(osp.join(pred_tag_path, name))

    pred = cPickle.load(open(osp.join(pred_path, name.replace("png", "pkl")), "rb"))
    pred_mask = np.argmax(pred, axis=2)
    pred_mask[0, 0] = 21

    crf_pred_mask = cv2.imread(osp.join(crf_pred_path, name))

    f = plt.figure(facecolor="white")
    ax = f.add_subplot(2, 3, 1)
    ax.imshow(im[:, :, ::-1])
    ax.axis("off")

    ax = f.add_subplot(2, 3, 2)
    ax.imshow(gt[:, :, ::-1])
    ax.axis("off")

    # set_trace()
    init_seed[0, 0] = 21
    ax = f.add_subplot(2, 3, 3)
    ax.imshow(init_seed, cmap=my_cmap)
    ax.axis("off")

    ax = f.add_subplot(2, 3, 4)
    ax.imshow(pred_mask, cmap=my_cmap)
    ax.axis("off")

    ax = f.add_subplot(2, 3, 5)
    ax.imshow(pred_tag[:, :, ::-1])
    ax.axis("off")

    ax = f.add_subplot(2, 3, 6)
    ax.imshow(crf_pred_mask[:, :, ::-1])
    ax.axis("off")
    
    # set_trace()
    score = im_score[name.split(".")[0]][0]
    title_str = "{:.4f}_{}".format(score, name.split(".")[0])

    plt.suptitle(title_str)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.savefig(os.path.join(save_path, name))
    plt.close()
