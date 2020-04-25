#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""
import findcaffe
import caffe
import argparse
import pprint
import numpy as np
import sys
import os
import pickle
sys.setrecursionlimit(1500)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from ipdb import set_trace
from scipy.ndimage import zoom

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--snapshot', dest='snapshot_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


class SolverWrapper(object):
    """A simple wrapper around Caffe'tools solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, pretrained_model=None, snapshot_model=None):
        """Initialize the SolverWrapper."""

        self.solver = caffe.SGDSolver(solver_prototxt)
        if snapshot_model is not None:
            print("restore from {}".format(snapshot_model))
            self.solver.restore(snapshot_model)
        elif pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

    def train_model(self):
        """Network training loop."""
        self.solver.solve()

        # modified by yaoqi, to debug the contrain loss layer
        # max_iter = 5
        # data = dict()
        # for k in range(max_iter):
        #     self.solver.step(1)
        #     data[str(k)] = dict()
            
        #     data[str(k)]["ims"] = self.solver.net.blobs["images"].data[...].copy()
        #     data[str(k)]["prob"] = self.solver.net.blobs["fc8-Softmax"].data[...].copy()
        #     data[str(k)]["loss"] = self.solver.net.blobs["loss-Constrain"].data[...].copy()
        #     print("data[{}][loss] = {}".format(k, data[str(k)]["loss"]))

        # for k in range(max_iter):
        #     print(data[str(k)]["loss"])
        # pickle.dump(data, open("constrain_loss_test_data.pkl", "wb"), pickle.HIGHEST_PROTOCOL)

        # modified by yaoqi, to debug the data layer
        # max_iter = 1000
        # for k in range(max_iter):
        #     self.solver.step(1)
        #     set_trace()
        #
        #     ims = self.solver.net.blobs["images"].data[...]
        #     mean_pixel = np.array([104.0, 117.0, 123.0])
        #     ims = ims + mean_pixel[None, :, None, None]
        #     ims = np.transpose(np.round(ims).astype(np.uint8), [0, 2, 3, 1])
        #
        #     seeds = np.transpose(self.solver.net.blobs["labels"].data[...], [0, 2, 3, 1])
        #
        #     probs = np.transpose(self.solver.net.blobs["fc8-Softmax"].data[...], [0, 2, 3, 1])
        #
        #     N = 4
        #     cols = 2
        #     f = plt.figure(facecolor="white")
        #     for ind in range(N):
        #         ax = f.add_subplot(N, cols, ind * cols + 1)
        #         seed = seeds[ind, :, :, 0]
        #
        #         # loc = np.where(seed > 0)
        #         # cue = np.ones(shape=(41, 41)) * 21
        #         # cue[loc[0], loc[1]] = loc[2]
        #         ax.matshow(seed)
        #         ax.axis("off")
        #
        #         ax = f.add_subplot(N, cols, ind * cols + 2)
        #         ax.imshow(ims[ind, :, :, ::-1])
        #         ax.axis("off")
        #
        #         # ax = f.add_subplot(N, cols, ind * cols + 3)
        #         # prob = np.argmax(probs[ind], axis=2)
        #         # prob[0, 0] = 0
        #         # prob[0, 1] = 21
        #         # ax.matshow(prob)
        #         # ax.axis("off")
        #     plt.tight_layout()
        #     plt.subplots_adjust(wspace=0.01, hspace=0.01)
        #     plt.show()

        # modified by yaoqi, to debug the SeedExpansion layer (cue in 41x41 shape)
        # max_iter = 1000
        # for k in range(max_iter):
        #     self.solver.net.forward()
        #     # if k <= 100:
        #     #     continue
        #     set_trace()
        #     ori_seeds = np.transpose(self.solver.net.blobs["labels"].data[...], [0, 2, 3, 1])
        #     new_seeds = np.transpose(self.solver.net.blobs["expand_seed"].data[...], [0, 2, 3, 1])
        #     probs = np.transpose(self.solver.net.blobs["fc8-Softmax"].data[...], [0, 2, 3, 1])
        #     sals = np.transpose(self.solver.net.blobs["sal"].data[...], [0, 2, 3, 1])
        #
        #     ims = self.solver.net.blobs["images"].data[...]
        #     mean_pixel = np.array([104.0, 117.0, 123.0])
        #     ims = zoom(ims, (1.0, 1.0, 41.0 / ims.shape[2], 41.0 / ims.shape[3]), order=1)
        #     ims = ims + mean_pixel[None, :, None, None]
        #     ims = np.transpose(np.round(ims).astype(np.uint8), [0, 2, 3, 1])
        #
        #     # N = ori_seeds.shape[0]
        #     N = 4
        #     cols = 5
        #     f = plt.figure(facecolor='white')
        #     for ind in range(N):
        #         ax = f.add_subplot(N, cols, ind * cols + 1)
        #         ori_seed = ori_seeds[ind, :, :, 0]
        #         ori_seed[0, 0] = 21
        #         ax.matshow(ori_seed)
        #         ax.axis("off")
        #
        #         ax = f.add_subplot(N, cols, ind * cols + 2)
        #         new_seed = new_seeds[ind, :, :, 0]
        #         new_seed[0, 0] = 21
        #         ax.matshow(new_seed)
        #         ax.axis("off")
        #
        #         ax = f.add_subplot(N, cols, ind * cols + 3)
        #
        #         im_labels = np.unique(ori_seed).tolist()
        #         im_non_labels = list(set([k for k in range(21)]).difference(set(im_labels)))
        #
        #         # prob = np.exp(log_probs[ind, :, :, :])
        #         prob = probs[ind, :, :, :]
        #         prob[:, :, im_non_labels] = 0
        #         res = np.argmax(prob, axis=2)
        #         max_prob = np.max(prob, axis=2)
        #         # res[max_prob < 0.99] = 21
        #         res[0, 0] = 21
        #         res[0, 1] = 0
        #         ax.matshow(res)
        #         ax.axis("off")
        #
        #         ax = f.add_subplot(N, cols, ind * cols + 4)
        #         im = ims[ind, :, :, :]
        #         ax.imshow(im[:, :, ::-1])
        #         ax.axis("off")
        #
        #         ax = f.add_subplot(N, cols, ind * cols + 5)
        #         sal = sals[ind, :, :, 0]
        #         ax.imshow(sal, cmap=cm.Greys_r)
        #         ax.axis("off")
        #
        #     plt.show()

        #modified by yaoqi, to debug the SeedExpansion layer (cue in 41x41x21 shape)
        # max_iter = 1000
        # for k in range(max_iter):
        #     self.solver.step(1)
        #     # if k < 20:
        #     #     continue
        #     set_trace()
        #
        #     ori_seeds = np.transpose(self.solver.net.blobs["labels"].data[...], [0, 2, 3, 1])
        #     new_seeds = np.transpose(self.solver.net.blobs["expand_seed"].data[...], [0, 2, 3, 1])
        #     log_probs = np.transpose(self.solver.net.blobs["fc8-CRF-log"].data[...], [0, 2, 3, 1])
        #
        #     ims = self.solver.net.blobs["images"].data[...]
        #     mean_pixel = np.array([104.0, 117.0, 123.0])
        #     ims = zoom(ims, (1.0, 1.0, 41.0 / ims.shape[2], 41.0 / ims.shape[3]), order=1)
        #     ims = ims + mean_pixel[None, :, None, None]
        #     ims = np.transpose(np.round(ims).astype(np.uint8), [0, 2, 3, 1])
        #
        #     # save_data = dict()
        #     # save_data["ori_seeds"] = ori_seeds
        #     # save_data["new_seeds"] = new_seeds
        #     # save_data["log_probs"] = log_probs
        #     # save_data["imgs"] = ims
        #     # pickle.dump(save_data, open(str(k) + ".pkl", "wb"))
        #
        #     N = 4
        #     cols = 4
        #     f = plt.figure(facecolor='white')
        #     for ind in range(N):
        #         ax = f.add_subplot(N, cols, ind * cols + 1)
        #         ori_seed = ori_seeds[ind]
        #         ori_loc = np.where(ori_seed > 0)
        #         ori_cue = np.ones(shape=(41, 41)) * 21
        #         ori_cue[ori_loc[0], ori_loc[1]] = ori_loc[2]
        #
        #         ax.matshow(ori_cue)
        #         ax.axis("off")
        #
        #         ax = f.add_subplot(N, cols, ind * cols + 2)
        #         new_seed = new_seeds[ind]
        #         new_loc = np.where(new_seed > 0)
        #         new_cue = np.ones(shape=(41, 41)) * 21
        #         new_cue[new_loc[0], new_loc[1]] = new_loc[2]
        #
        #         ax.matshow(new_cue)
        #         ax.axis("off")
        #
        #         ax = f.add_subplot(N, cols, ind * cols + 3)
        #         im_labels = np.unique(new_loc[2]).tolist()
        #         im_non_labels = list(set([k for k in range(21)]).difference(set(im_labels)))
        #
        #         prob = np.exp(log_probs[ind, :, :, :])
        #         prob[:, :, im_non_labels] = 0
        #         res = np.argmax(prob, axis=2)
        #         max_prob = np.max(prob, axis=2)
        #         # res[max_prob < 0.99] = 21
        #         res[0, 0] = 21
        #         res[0, 1] = 0
        #         ax.matshow(res)
        #         ax.axis("off")
        #
        #         ax = f.add_subplot(N, cols, ind * cols + 4)
        #         im = ims[ind, :, :, :]
        #         ax.imshow(im[:, :, ::-1])
        #         ax.axis("off")
        #
        #     plt.show()

class Arg(object):
    pass


if __name__ == '__main__':
    args = parse_args()

    # set up caffe
    caffe.set_mode_gpu()
    if args.gpu_id is not None:
        caffe.set_device(args.gpu_id)

    sw = SolverWrapper(args.solver,
                       pretrained_model=args.pretrained_model, snapshot_model=args.snapshot_model)

    print 'Solving...'
    sw.train_model()
    print 'done solving'
