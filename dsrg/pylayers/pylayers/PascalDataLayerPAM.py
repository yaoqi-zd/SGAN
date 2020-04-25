from __future__ import print_function
from __future__ import division

import json
import time
import pickle
from scipy.ndimage import zoom
import cv2
import caffe
import math
from ipdb import set_trace

import numpy as np
import os.path as osp

from random import shuffle
import random


class PascalDataLayerPAM(caffe.Layer):
    """
    This is a simple data layer for training a weakly segmentation network with seed
    """

    def setup(self, bottom, top):

        #============== Read input parameters ==============#
        # params is a python dictionary with layer parameters
        params = eval(self.param_str)

        MyTransformer.check_params(params)

        self.batch_size = params["batch_size"]
        self.input_shape = params["new_size"]

        # create a batch loader to load the images
        self.batch_loader = BatchLoader(params)

        #============ Reshape tops (im, label, sim_map) ===================#
        feat_size = int(round(self.input_shape[0] / 8 + 0.5))
        top[0].reshape(self.batch_size, 3, self.input_shape[0], self.input_shape[1])
        top[1].reshape(self.batch_size, 20)
        top[2].reshape(self.batch_size, feat_size*feat_size, feat_size*feat_size)

    def forward(self, bottom, top):
        """
        Load data
        """
        for itt in range(self.batch_size):
            # use the batch loader to load the next image.
            im, label, sim_map = self.batch_loader.load_next_image()

            # add directly to the caffe data layer
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = 0.0
            top[1].data[itt, label - 1] = 1.0
            top[2].data[itt, ...] = sim_map

    def reshape(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass


class BatchLoader(object):

    def __init__(self, params):
        self.batch_size = params["batch_size"]
        self.root_folder = params["root_folder"] # VOC2012 folder
        self.cues_name = params["cue_name"]
        self.source = params["source"]
        np.random.seed(0)
        random.seed(0)

        # get list of image indexes
        self.indexlist = [line.strip().split(' ') for line in open(self.source, "r")]
        self._cur = 0

        self.transformer = MyTransformer(params)

        self.loc_cues = pickle.load(open(self.cues_name, "rb"))

        print("BatchLoader initialized with {} images".format(len(self.indexlist)))

    def load_next_image(self):
        """
        Load the next image in a batch
        """
        # Do we finish an epoch?
        if self._cur == len(self.indexlist):
            self._cur = 0
            print("-----------------------reshuffling----------------------")
            shuffle(self.indexlist)

        # set_trace()

        # Load an image
        index, cue_ind = self.indexlist[self._cur]
        image_file_path = osp.join(self.root_folder, "JPEGImages", str(index))
        image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)

        label = self.loc_cues[str(cue_ind) + "_labels"]

        sal_full_path = osp.join(self.root_folder, "sal_sdnet", str(index).replace("jpg", "png"))
        sal = cv2.imread(sal_full_path, cv2.IMREAD_GRAYSCALE)

        image, sim_map = self.transformer.preprocess(image, sal)

        self._cur += 1
        return image, label, sim_map


class MyTransformer(object):

    def __init__(self, params):
        MyTransformer.check_params(params)
        self.mean = params["mean"]
        self.is_mirror = params["mirror"]
        self.new_h, self.new_w = params["new_size"]

    def set_mean(self, mean):
        self.mean = mean

    def preprocess(self, image, sal):

        # resize
        img_h, img_w = image.shape[:2]
        image = zoom(image, (self.new_h / img_h, self.new_w / img_w, 1.0), order=1)
        sal = zoom(sal, (self.new_h / img_h, self.new_w / img_w), order=1)

        # subtract mean
        image = image.astype(np.float32)
        image -= self.mean

        # for sal from sdnet
        sal = np.asarray(sal, np.float32)
        sal /= 255
        sal_bi = np.ones(sal.shape) * 21
        sal_bi[sal >= 0.2] = 1
        sal_bi[sal <= 0.06] = 0
        sal_bi = sal_bi.astype(np.uint8)

        # HWC->CHW
        image = image.transpose((2, 0, 1))

        # random flip
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            sal_bi = sal_bi[:, ::flip]

        # for sal to compute similarity map
        feat_size = int(round(self.new_h / 8 + 0.5))
        sal_reduced = zoom(sal_bi, (feat_size / self.new_h, feat_size / self.new_h), order=0)
        sal_reshape = np.tile(np.reshape(sal_reduced, (feat_size * feat_size, -1)), (1, feat_size * feat_size))
        sal_reshape_1 = np.transpose(sal_reshape)
        sim_map = np.equal(sal_reshape, sal_reshape_1)
        sim_map = sim_map.astype(np.float32)

        sim_map = sim_map / (np.sum(sim_map, axis=1, keepdims=True) + 1e-5)

        return image, sim_map

    @classmethod
    def check_params(cls, params):
        if "new_size" not in params:
            params["new_size"] = (321, 321)
        if "mean" not in params:
            params["mean"] = (104.008,116.669,122.675)
        if "mirror" not in params:
            params["mirror"] = True
        if "source" not in params:
            params["source"] = 'input_list.txt'
        if "cue_name" not in params:
            params['cue_name'] = 'localization_cues-sal.pickle'

        assert osp.exists(params["cue_name"])
        assert osp.exists(params["root_folder"])
        assert osp.exists(params["source"])


if __name__ == '__main__':
    params = {"batch_size": 1,
              "mean": np.array([0.0, 0.0, 0.0]),
              "root_folder": "/data1/yaoqi/Dataset/VOCdevkit/VOC2012/",
              "source":"/data1/yaoqi/segmentation/weakly/DSRG-master/training/experiment/seed_mc/list/train_aug_id.txt",
              "mirror": True,
              "crop_size": (321, 321),
              "new_size": (353, 353),
              "cue_name": "/data1/yaoqi/segmentation/weakly/DSRG-master/training/localization_cues/localization_cues-sal.pickle"}
    t = MyTransformer(params)

    # cue_name = "/data1/yaoqi/segmentation/weakly/DSRG-master/training/localization_cues/generate_seed/my_localization_cues.pkl"

    cue_id = "6154" #"479"
    im_name = "2007_000032"#"2007_000039"
    cues = pickle.load(open(params["cue_name"], "rb"))
    # cue = cues["2007_000039_cues"]
    cue = cues[cue_id + "_cues"]
    seed = np.zeros(shape=(41, 41, 21), dtype=np.float32)
    seed[cue[1], cue[2], cue[0]] = 1.0
    image = cv2.imread(osp.join(params["root_folder"], "JPEGImages", im_name + ".jpg"))

    for _ in range(100):
        crop_im, new_seed = t.preprocess(image, seed)
        crop_im = crop_im.transpose((1, 2, 0)).astype(np.uint8)
        new_seed = new_seed.transpose((1, 2, 0))

        loc = np.where(new_seed == 1)
        mask = np.ones(shape=(41, 41)) * 21
        mask[loc[0], loc[1]] = loc[2]

        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        f1 = plt.figure(facecolor="white")
        rows, cols = 1, 2
        ax = f1.add_subplot(rows, cols, 1)
        ax.imshow(crop_im[:, :, ::-1])
        ax.axis("off")

        ax = f1.add_subplot(rows, cols, 2)
        ax.matshow(mask)
        ax.axis("off")

        # ax = f1.add_subplot(rows, cols, 3)
        # ax.imshow(new_sal[:, :, 0], cmap=cm.Greys_r)
        # ax.axis("off")
        plt.show()

        set_trace()









