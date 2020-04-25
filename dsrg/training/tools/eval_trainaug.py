from __future__ import print_function
from __future__ import division

import findcaffe
import caffe

import os, argparse
import os.path as osp
import numpy as np
import scipy.ndimage as nd
import pylab, png, pickle
from shutil import copyfile

from ipdb import set_trace
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import matplotlib.colors as mpl_colors
import cv2
palette = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.5, 0.0),
           (0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5), (0.5, 0.5, 0.5),
           (0.25, 0.0, 0.0), (0.75, 0.0, 0.0), (0.25, 0.5, 0.0), (0.75, 0.5, 0.0),
           (0.25, 0.0, 0.5), (0.75, 0.0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5),
           (0.0, 0.25, 0.0), (0.5, 0.25, 0.0), (0.0, 0.75, 0.0), (0.5, 0.75, 0.0),
           (0.0, 0.25, 0.5), (0.75, 0.75, 0.75)]
my_cmap = mpl_colors.LinearSegmentedColormap.from_list('Custom cmap', palette, 22)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_file", default=None, type=str)
    parser.add_argument("--weight_file", default=None, type=str)
    parser.add_argument("--list_file", default=None, type=str)
    parser.add_argument("--save_path", default=None, type=str)
    args = parser.parse_args()

    return args

def preprocess(image, label, size, mean_pixel):
    image = nd.zoom(image.astype('float32'),
                    (size / float(image.shape[0]),
                     size / float(image.shape[1]), 1.0),
                    order=1)
    label = nd.zoom(label, 
                    (size / float(label.shape[0]), 
                    size / float(label.shape[1])), 
                    order=0)

    # image = image[:, :, [2, 1, 0]]
    image = image - mean_pixel

    image = image.transpose([2, 0, 1])
    image = np.expand_dims(image, 0)

    label = np.reshape(label, newshape=(1, 1, size, size))
    return image, label


class CaffeInfer(object):
    def __init__(self, net_file, weight_file, gpu_id=0):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()
        self.net = caffe.Net(args.net_file, args.weight_file, caffe.TRAIN)

    def eval(self, list_file, save_viz_path, save_score_name):
        train_samples = [line.strip().split(" ") for line in open(list_file, "r").readlines()]
        mean_pixel = np.array([104.0, 117.0, 123.0])

        im_score = dict()

        for k, (im_name, label_name) in enumerate(train_samples):
            raw_im, raw_label = cv2.imread(im_name), cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
            im, label = preprocess(raw_im, raw_label, 321, mean_pixel)
            
            self.net.blobs['images'].data[...] = im
            self.net.blobs['labels'].data[...] = label
            self.net.forward()

            score = np.squeeze(self.net.blobs['fc8_merge'].data[...])
            score = nd.zoom(score, (1, raw_im.shape[0] / score.shape[1], raw_im.shape[1] / score.shape[2]), order=1)
            pred = np.argmax(score, axis=0)

            loss_seed = self.net.blobs['loss-Seed'].data[...][0]
            loss_constrain = self.net.blobs['loss-Constrain'].data[...][0]

            # draw result
            # f = plt.figure(facecolor="white")
            # ax = f.add_subplot(2, 2, 1)
            # ax.imshow(raw_im[:, :, ::-1])
            # ax.axis("off")

            # raw_label[0, 0] = 21
            # ax = f.add_subplot(2, 2, 2)
            # ax.imshow(raw_label, cmap=my_cmap)
            # ax.axis("off")
            
            # pred[0, 0] = 21
            # ax = f.add_subplot(2, 2, 3)
            # ax.imshow(pred, cmap=my_cmap)
            # ax.axis("off")

            # gt_name = im_name.replace("JPEGImages", "SegmentationClassAug_color").replace("jpg", "png")
            # gt = cv2.imread(gt_name)
            # ax = f.add_subplot(2, 2, 4)
            # ax.imshow(gt[:, :, ::-1])
            # ax.axis("off")

            pure_name = im_name.split("/")[-1].split(".")[0]
            # title_str = "{:s}_{:.2f}_{:.2f}".format(pure_name, loss_seed, loss_constrain)
            # plt.suptitle(title_str)
            # plt.tight_layout()
            # plt.subplots_adjust(wspace=0.01, hspace=0.01)
            # plt.savefig(os.path.join(save_viz_path, pure_name + ".png"))
            # plt.close()

            im_score[pure_name] = [loss_seed, loss_constrain]

            print("finish {} files".format(k))

        pickle.dump(im_score, open(save_score_name, "wb"), pickle.HIGHEST_PROTOCOL)

def cmp(x, y):
    res = 1 if x[1][0] < y[1][0] else 0
    return res

if __name__ == "__main__":
    args = parse_args()
    root_path = "/data1/yaoqi/segmentation/weakly/wsss/dsrg/training/experiment/anti-noise/"
    args.net_file = osp.join(root_path, "config/deeplabv2_weakly_forward.prototxt")
    args.weight_file = osp.join(root_path, "model/model-sgan_iter_8000.caffemodel")
    args.list_file = osp.join(root_path, "list/train_aug_pt_0328_ratio15.txt")
    args.save_path = "training/visualize/eval_train"
    save_score_name = "training/visualize/im_score_6396.pkl"

    # infer im-score viz results if not exists
    # if not osp.exists(args.save_path):
    #     os.makedirs(args.save_path)
    #     caffe_infer = CaffeInfer(args.net_file, args.weight_file)
    #     caffe_infer.eval(args.list_file, args.save_path, save_score_name)
    
    # find top loss and least loss train samples
    im_score = pickle.load(open(save_score_name, "rb"))
    sort_im_score = sorted(im_score.items(), key=lambda d: d[1][0])
    
    # top_loss_path = "visualize/top_100_loss"
    # least_loss_path = "visualize/least_100_loss"
    # if not osp.exists(top_loss_path):
    #     os.makedirs(top_loss_path)
    # if not osp.exists(least_loss_path):
    #     os.makedirs(least_loss_path)

    # for k in range(100):
    #     least_name = sort_im_score[k][0] + ".png"
    #     top_name = sort_im_score[-(k + 1)][0] + ".png"
    #     copyfile(osp.join(args.save_path, least_name), osp.join(least_loss_path, least_name))
    #     copyfile(osp.join(args.save_path, top_name), osp.join(top_loss_path, top_name))

    write_im_path = "/home/yaoqi/Dataset/VOC2012/JPEGImages/"
    write_gt_path = "/data1/yaoqi/home_segmentation/segmentation/weakly/DSRG/training/localization_cues/seed_0328_7440_5996/"
    with open("train_aug_pt_1012_retrain_10282.txt", "w") as f:
        for num, (k, v) in enumerate(sort_im_score):
            if num < 10282:
                f.write("{} {}\r\n".format(write_im_path + k + ".jpg", write_gt_path + k + ".png"))


    # sort_path = "visualize/sort_eval_train"
    # # set_trace()
    # if not osp.exists(sort_path):
    #     os.makedirs(sort_path)
    # for k, v in sort_im_score:
    #     save_im_name = "{:.3f}_{:s}".format(v[0], k)
    #     copyfile(osp.join(args.save_path, k + ".png"), osp.join(sort_path, save_im_name + ".png"))
    
    
