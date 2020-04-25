from __future__ import print_function
from __future__ import division

import caffe

import numpy as np
from scipy.ndimage import zoom

import theano
import theano.tensor as T

import cPickle
import yaml
import cv2
import os.path as osp
import multiprocessing
from numpy.random import shuffle

from krahenbuhl2013 import CRF
import CC_labeling_8
from sklearn.cluster import KMeans

min_prob = 0.0001
R = 5

class Label2Seed(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 1:
            raise Exception("Need one input")

    def reshape(self, bottom, top):
        n, _, h, w = bottom[0].data[...].shape
        top[0].reshape(n, 21, h, w)

    def forward(self, bottom, top):
        batch_size, _, h, w = bottom[0].data[...].shape
        labels = np.transpose(bottom[0].data[...], [0, 2, 3, 1])

        seeds = np.zeros(shape=(batch_size, h, w, 21), dtype=np.float32)
        for k in range(batch_size):
            label = labels[k, :, :, 0]
            cor = np.where(label != 21)
            # set_trace()
            seeds[k, cor[0], cor[1], label[cor].astype(np.int16)] = 1.0

        top[0].data[...] = np.transpose(seeds, [0, 3, 1, 2])

    def backward(self, top, prop_down, bottom):
        bottom[0].diff[...] = 0.0

class Seed2Label(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 1:
            raise Exception("Need one input")

    def reshape(self, bottom, top):
        batch_size, _, h, w = bottom[0].data[...].shape
        top[0].reshape(batch_size, 1, h, w)

    def forward(self, bottom, top):
        batch_size, _, h, w = bottom[0].data[...].shape
        seeds = np.transpose(bottom[0].data[...], [0, 2, 3, 1])

        labels = np.ones(shape=(batch_size, h, w, 1)) * 21
        for k in range(batch_size):
            seed = seeds[k]
            cor = np.where(seed == 1)
            labels[k, cor[0], cor[1], 0] = cor[2]

        top[0].data[...] = np.transpose(labels, [0, 3, 1, 2])

    def backward(self, top, prop_down, bottom):
        bottom[0].diff[...] = 0.0


class SoftmaxLayer(caffe.Layer):

    def setup(self, bottom, top):

        if len(bottom) != 1:
            raise Exception("Need two inputs to compute distance.")

        preds = T.ftensor4()
        top_diff = T.ftensor4()

        preds_max = T.addbroadcast(T.max(preds, axis=1, keepdims=True), 1)
        preds_exp = np.exp(preds - preds_max)
        probs = preds_exp / T.addbroadcast(T.sum(preds_exp, axis=1, keepdims=True), 1) + min_prob
        probs = probs / T.sum(probs, axis=1, keepdims=True)

        probs_sum = T.sum(probs * top_diff)

        self.forward_theano = theano.function([preds], probs)
        self.backward_theano = theano.function([preds, top_diff], T.grad(probs_sum, preds))

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        top[0].data[...] = self.forward_theano(bottom[0].data[...])

    def backward(self, top, prop_down, bottom):
        grad = self.backward_theano(bottom[0].data[...], top[0].diff[...])
        bottom[0].diff[...] = grad


class PAM(caffe.Layer):
    def setup(self, bottom, top):

        if len(bottom) != 2:
            raise Exception("PAM layer needs two inputs")

        conv5 = T.ftensor4()
        sim_map = T.ftensor3()
        top_diff = T.ftensor4()

        batch_size, c, h, w = conv5.shape
        value = T.reshape(conv5, newshape=(batch_size, c, h * w))
        value = T.transpose(value, axes=(0, 2, 1))

        context = T.batched_dot(sim_map, value)
        context = T.transpose(context, axes=(0, 2, 1))
        context = T.reshape(context, newshape=(batch_size, c, h, w))

        fuse = context + conv5

        fuse_sum = T.sum(fuse * top_diff)

        self.forward_theano = theano.function([conv5, sim_map], fuse)
        self.backward_theano = theano.function([conv5, sim_map, top_diff], T.grad(fuse_sum, conv5))

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        top[0].data[...] = self.forward_theano(bottom[0].data[...], bottom[1].data[...])

    def backward(self, top, prop_down, bottom):
        grad = self.backward_theano(bottom[0].data[...], bottom[1].data[...], top[0].diff[...])
        bottom[0].diff[...] = grad


class CRFLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("The layer needs two inputs!")

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):

        probs = bottom[0].data
        _, _, h, w = probs.shape
        probs[probs < min_prob] = min_prob
        unary = np.transpose(np.array(probs), [0, 2, 3, 1])

        mean_pixel = np.array([104.0, 117.0, 123.0])
        im = bottom[1].data[...]
        im = zoom(im, (1.0, 1.0, float(h) / im.shape[2], float(w) / im.shape[3]), order=1)
        im = np.transpose(im, [0, 2, 3, 1])
        im = im + mean_pixel[None, None, None, :]
        im = np.round(im)

        N = unary.shape[0]

        self.result = np.zeros(unary.shape)

        for i in range(N):
            self.result[i] = CRF(im[i], unary[i], scale_factor=12.0)

        self.result = np.transpose(self.result, [0, 3, 1, 2])
        self.result[self.result < min_prob] = min_prob
        self.result = self.result / np.sum(self.result, axis=1, keepdims=True)

        top[0].data[...] = np.log(self.result)

    def backward(self, top, prop_down, bottom):
        grad = (1 - self.result) * top[0].diff[...]
        bottom[0].diff[...] = grad
        # bottom[0].diff[...] = 0.0

class SeedLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("The layer needs two inputs!")

        probs = T.ftensor4()
        labels = T.ftensor4()

        count = T.sum(labels, axis=(1, 2, 3), keepdims=True)
        loss_balanced = -T.mean(T.sum(labels * T.log(probs), axis=(1, 2, 3), keepdims=True) / T.maximum(count, min_prob))

        self.forward_theano = theano.function([probs, labels], loss_balanced)
        self.backward_theano = theano.function([probs, labels], T.grad(loss_balanced, probs))

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        top[0].data[...] = self.forward_theano(bottom[0].data[...], bottom[1].data[...])

    def backward(self, top, prop_down, bottom):
        grad = self.backward_theano(bottom[0].data[...], bottom[1].data[...])
        bottom[0].diff[...] = grad

class BalancedSeedLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("The layer needs two inputs!")

        probs = T.ftensor4()
        labels = T.ftensor4()

        probs_bg = probs[:, 0, :, :]
        labels_bg = labels[:, 0, :, :]
        probs_fg = probs[:, 1:, :, :]
        labels_fg = labels[:, 1:, :, :]

        count_bg = T.sum(labels_bg, axis=(1, 2), keepdims=True)
        count_fg = T.sum(labels_fg, axis=(1, 2, 3), keepdims=True)
        loss_1 = -T.mean(T.sum(labels_bg * T.log(probs_bg), axis=(1, 2), keepdims=True) / T.maximum(count_bg, min_prob))
        loss_2 = -T.mean(T.sum(labels_fg * T.log(probs_fg), axis=(1, 2, 3), keepdims=True) / T.maximum(count_fg, min_prob))

        loss_balanced = loss_1 + loss_2

        self.forward_theano = theano.function([probs, labels], loss_balanced)
        self.backward_theano = theano.function([probs, labels], T.grad(loss_balanced, probs))

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        top[0].data[...] = self.forward_theano(bottom[0].data[...], bottom[1].data[...])

    def backward(self, top, prop_down, bottom):
        grad = self.backward_theano(bottom[0].data[...], bottom[1].data[...])
        bottom[0].diff[...] = grad

class ConstrainLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("The layer needs two inputs!")

        probs = T.ftensor4()
        probs_smooth_log = T.ftensor4()

        probs_smooth = T.exp(probs_smooth_log)

        loss = T.mean(T.sum(probs_smooth * T.log(T.clip(probs_smooth / probs, 0.05, 20)), axis=1)) #

        self.forward_theano = theano.function([probs, probs_smooth_log], loss)
        self.backward_theano = theano.function([probs, probs_smooth_log], T.grad(loss, [probs, probs_smooth_log]))

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        top[0].data[...] = self.forward_theano(bottom[0].data[...], bottom[1].data[...])

    def backward(self, top, prop_down, bottom):
        grad = self.backward_theano(bottom[0].data[...], bottom[1].data[...])[0]
        bottom[0].diff[...] = grad
        grad = self.backward_theano(bottom[0].data[...], bottom[1].data[...])[1]
        bottom[1].diff[...] = grad


##########################################SEED EXPANSION LAYER################################
eps = 0.00001
seed_ratio = 1000
prob_conf_thresh = 0.99
bg_thresh = 0.5
fg_thresh = 0.7

from ipdb import set_trace

def collapse_seed_to_cue(seed):
    h, w = seed.shape[:2]
    cue = np.ones(shape=(h, w)) * 21
    loc = np.where(seed > 0)
    cue[loc[0], loc[1]] = loc[2]

    img_tags = np.unique(loc[2])

    return cue, img_tags

def set_seed_unary(seed, num_labels):
    """
    set seed unary term
    :param seed: eg. 41x41
    :param num_labels: eg. 21
    :return: seed unary: 41x41x21
    """
    h, w = seed.shape[:2]
    seed_unary = np.zeros(shape=(h, w, num_labels))

    sure_ind = np.where(seed < num_labels)
    sure_value = seed[sure_ind]
    seed_unary[sure_ind[0], sure_ind[1], sure_value] = 1.0

    return seed_unary

def set_seed_unary_21(seed):
    """
    set seed unary term
    :param seed: 41x41x21, value range [0, 1]
    :return: seed unary: 41x41x21
    """
    seed_unary = seed / (np.sum(seed, axis=2, keepdims=True) + eps)
    return seed_unary

def expand_seed_21(ini_seed, ini_prob, im, expand_bkg):
    """
    expand seed when seed is of shape (41, 41, 21)
    """
    loc = np.where(ini_seed > 0)
    im_labels = np.unique(loc[2]).tolist()
    im_non_labels = list(set([k for k in range(21)]).difference(set(im_labels)))

    res_unary = np.argmax(ini_prob, axis=2)
    unary_zeros_ratio = len(np.where(res_unary == 0)[0]) / res_unary.size

    ini_cue, _ = collapse_seed_to_cue(ini_seed)

    if unary_zeros_ratio > 0.95:
        # return np.expand_dims(ini_cue, axis=2)
        return ini_seed

    seed_unary = set_seed_unary_21(ini_seed)
    seed_unary[seed_unary < eps] = eps
    unary = seed_ratio * seed_unary + ini_prob
    unary = unary / (np.sum(unary, axis=2, keepdims=True) + eps)

    crf_prob = CRF(im, np.log(unary), scale_factor=12.0)
    crf_prob[:, :, im_non_labels] = 0

    # ############################ generate new seed #########################
    # new_seed = np.zeros(shape=ini_seed.shape, dtype=np.float32)
    # for cls in im_labels:
    #     thresh = bg_thresh if cls == 0 else fg_thresh
    #     cls_prob = crf_prob[:, :, cls]
    #     if cls != 0:
    #         new_seed[:, :, cls] = cls_prob > thresh * np.max(cls_prob)
    #     new_seed[:, :, 0] = ini_seed[:, :, 0]
    # ############################ generate new seed #########################

    res_crf = np.argmax(crf_prob, axis=2)
    max_prob = np.max(crf_prob, axis=2)
    res_crf[max_prob < prob_conf_thresh] = 21

    new_seed = np.copy(ini_seed)
    for v in sorted(im_labels):
        if v == 21:
            continue
        if v == 0:
            if not expand_bkg:
                continue

        ind = np.where(res_crf == v)
        new_seed[ind[0], ind[1], v] = 1.0
    # return np.expand_dims(new_seed, axis=2)
    return new_seed

def expand_seed(ini_seed, ini_prob, im):
    """
    expand seed when seed is  of shape (41, 41), range [0, 21]
    """
    assert ini_seed.shape[:2] == (41, 41)
    assert ini_prob.shape == (41, 41, 21)
    assert im.shape == (41, 41, 3)

    im_labels = np.unique(ini_seed).tolist()
    im_non_labels = list(set([k for k in range(21)]).difference(set(im_labels)))

    # set_trace()

    ################################ check if unary gives all bkg prediction ########################
    res_unary = np.argmax(ini_prob, axis=2)
    unary_zeros_ratio = len(np.where(res_unary == 0)[0]) / res_unary.size
    # seed_non_zeros_ratio = len(np.where((ini_seed > 0) & (ini_seed < 21))[0]) / ini_seed.size

    if unary_zeros_ratio > 0.95:
        return np.expand_dims(ini_seed, axis=2)
    ################################ check if unary gives all bkg prediction ########################

    ################################ expand seed with CRF ########################
    seed_unary = set_seed_unary(ini_seed, num_labels=21)
    seed_unary[seed_unary < eps] = eps

    unary = seed_ratio * seed_unary + ini_prob
    unary = unary / (np.sum(unary, axis=2, keepdims=True) + eps)

    crf_prob = CRF(im, np.log(unary), scale_factor=12.0)
    crf_prob[:, :, im_non_labels] = 0

    res_crf = np.argmax(crf_prob, axis=2)
    max_prob = np.max(crf_prob, axis=2)
    res_crf[max_prob < prob_conf_thresh] = 21
    ################################ expand seed with CRF ########################

    ################################ do not update bkg ########################
    new_seed = np.copy(ini_seed)
    unique_v = np.unique(ini_seed).tolist()
    for v in unique_v:
        if v == 21:
            continue
        # if v == 0:  # do not update bkg
        #     continue
        ind = np.where(res_crf == v)
        new_seed[ind[0], ind[1]] = v
    ################################ do not update bkg ########################

    return np.expand_dims(new_seed, axis=2)

class SeedExpansionLayer(caffe.Layer):
    ## bottom: 0-seed, 1-prob(before CRF), 2-im
    ## top: 0-new_seed
    def setup(self, bottom, top):
        if len(bottom) != 3:
            raise Exception("SeedExpansionLayer need three inputs!")
        self.iter = 0
        params = eval(self.param_str)
        self.start_iter = params["start_iter"]

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        self.iter += 1
        if self.iter >= self.start_iter:  # default 100
            # print("start expanding iter:{}".format(self.iter))
            unary = np.transpose(np.array(bottom[1].data[...]), [0, 2, 3, 1])
            # unary = np.exp(unary) # since we use prob before log(CRF()) now, comment out this line

            mean_pixel = np.array([104.008, 116.669, 122.675])
            im = bottom[2].data[...]
            im = zoom(im, (1.0, 1.0, 41.0 / im.shape[2], 41.0 / im.shape[3]), order=1)
            im = im + mean_pixel[None, :, None, None]
            im = np.transpose(np.round(im).astype(np.uint8), [0, 2, 3, 1])

            seed = np.transpose(np.array(bottom[0].data[...]), [0, 2, 3, 1]).astype(np.uint8)

            N = unary.shape[0]
            result = np.zeros(seed.shape)

            for i in range(N):

                result[i] = expand_seed(seed[i, :, :, 0], unary[i], im[i])

            result = np.transpose(result, [0, 3, 1, 2])

            top[0].data[...] = result
        else:
            top[0].data[...] = bottom[0].data[...]

    def backward(self, top, prop_down, bottom):
        bottom[1].diff[...] = 0.0

##########################################SEED EXPANSION LAYER################################


class ExpandLossLayer(caffe.Layer):

    def setup(self, bottom, top):

        if len(bottom) != 2:
            raise Exception("The layer needs two inputs!")

        probs_tmp = T.ftensor4()
        stat_inp = T.ftensor4()

        stat = stat_inp[:, :, :, 1:]

        probs_bg = probs_tmp[:, 0, :, :]
        probs = probs_tmp[:, 1:, :, :]

        probs_max = T.max(probs, axis=3).max(axis=2)

        q_fg = 0.996
        probs_sort = T.sort(probs.reshape((-1, 20, 41 * 41)), axis=2)
        weights = np.array([q_fg ** i for i in range(41 * 41 - 1, -1, -1)])[None, None, :]
        Z_fg = np.sum(weights)
        weights = T.addbroadcast(theano.shared(weights), 0, 1)
        probs_mean = T.sum((probs_sort * weights) / Z_fg, axis=2)

        q_bg = 0.999
        probs_bg_sort = T.sort(probs_bg.reshape((-1, 41 * 41)), axis=1)
        weights_bg = np.array([q_bg ** i for i in range(41 * 41 - 1, -1, -1)])[None, :]
        Z_bg = np.sum(weights_bg)
        weights_bg = T.addbroadcast(theano.shared(weights_bg), 0)
        probs_bg_mean = T.sum((probs_bg_sort * weights_bg) / Z_bg, axis=1)

        stat_2d = stat[:, 0, 0, :] > 0.5

        loss_1 = -T.mean(T.sum((stat_2d * T.log(probs_mean) / T.sum(stat_2d, axis=1, keepdims=True)), axis=1))
        loss_2 = -T.mean(T.sum(((1 - stat_2d) * T.log(1 - probs_max) / T.sum(1 - stat_2d, axis=1, keepdims=True)), axis=1))
        loss_3 = -T.mean(T.log(probs_bg_mean))

        loss = loss_1 + loss_2 + loss_3

        self.forward_theano = theano.function([probs_tmp, stat_inp], loss)
        self.backward_theano = theano.function([probs_tmp, stat_inp], T.grad(loss, probs_tmp))

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        top[0].data[...] = self.forward_theano(bottom[0].data[...], bottom[1].data[...])

    def backward(self, top, prop_down, bottom):
        grad = self.backward_theano(bottom[0].data[...], bottom[1].data[...])
        bottom[0].diff[...] = grad

import timeit
from multiprocessing import Pool
def generate_seed_step(item):
    labels, seed_c, probs_refinement, th1, th2 = item

    cls_index = np.where(labels == 1)[0]
    probs_selected = probs_refinement[cls_index]
    probs_c = np.argmax(probs_selected, axis=0)
    probs_p = np.max(probs_selected, axis=0)

    channels, height, width = seed_c.shape
    label_map = np.zeros((height, width))

    index1 = np.where(seed_c > 0)

    label_map[index1[1], index1[2]] = index1[0] + 1 # 1-index
    for (x,y), value in np.ndenumerate(probs_p):
        c = cls_index[probs_c[x,y]]
        if value > th2:
            if not c == 0:
                label_map[x, y] = c + 1
            elif value > th1:
                label_map[x, y] = c + 1

    for c in cls_index:
        mat = (label_map == (c+1))
        mat = mat.astype(int)
        cclab = CC_labeling_8.CC_lab(mat) # Our reviewer suggests to use connected component for accelaration, Thanks!
        cclab.connectedComponentLabel()
        high_confidence_set_label = set()
        for (x,y), value in np.ndenumerate(mat):
            if value == 1 and seed_c[c, x, y] == 1:
                high_confidence_set_label.add(cclab.labels[x][y])
            elif value == 1 and np.sum(seed_c[:, x, y]) == 1:
                cclab.labels[x][y] = -1
        for (x,y), value in np.ndenumerate(np.array(cclab.labels)):
            if value in high_confidence_set_label:
                seed_c[c, x, y] = 1

    return seed_c

class DSRGLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 4:
            raise Exception("The layer needs four inputs!")

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)
        # self._do_save = layer_params['do_save']
        self._th1 = layer_params['th1']
        self._th2 = layer_params['th2']
        if 'iters' not in layer_params:
            layer_params['iters'] = -1
        self._max_iters = layer_params['iters']
        self._iter_index = 0
        self.pool = Pool()

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[1].data.shape)

    def forward(self, bottom, top):
        img_labels, probs, cues, im = bottom[0].data, bottom[1].data, bottom[2].data, bottom[3].data
        num, channels, height, width = probs.shape
        seed_c = self.generate_seed(img_labels, probs, cues, im)

        self._iter_index = self._iter_index + 1
        top[0].data[...] = seed_c


    def backward(self, top, prop_down, bottom):
        bottom[1].diff[...] = top[0].diff

    def refinement(self, probs, im, scale_factor=12.0):
        _, _, h, w = probs.shape
        probs[probs < min_prob] = min_prob
        unary = np.transpose(np.array(probs), [0, 2, 3, 1])

        mean_pixel = np.array([104.0, 117.0, 123.0])
        im = zoom(im, (1.0, 1.0, float(h) / im.shape[2], float(w) / im.shape[3]), order=1)
        im = np.transpose(im, [0, 2, 3, 1])
        im = im + mean_pixel[None, None, None, :]
        im = np.round(im)

        N = unary.shape[0]

        result = np.zeros(unary.shape)

        for i in range(N):
            result[i] = CRF(im[i], unary[i], scale_factor=scale_factor)

        result = np.transpose(result, [0, 3, 1, 2])
        result[result < min_prob] = min_prob
        result = result / np.sum(result, axis=1, keepdims=True)
        return result

    def generate_seed(self, labels, probs, cues, im):
        num, channels, height, width = probs.shape
        probs_refinement = self.refinement(probs, im, 12.0)
        # probs_refinement = np.exp(probs)

        seed_c = np.zeros_like(probs)
        seed_c[...] = cues

        items_for_map = [[labels[batch_id, 0, 0], seed_c[batch_id], probs_refinement[batch_id], self._th1, self._th2] for batch_id in xrange(num)]
        # seed_c_all = self.pool.map(generate_seed_step, items_for_map)
        seed_c_all = []
        for item in items_for_map:
            seed_c_all.append(generate_seed_step(item))

        return np.array(seed_c_all)

class ClsAnnotationLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 1:
            raise  Exception("The layer needs only one inputs!")

        params = eval(self.param_str)
        if "cue_path" not in params:
            params["cue_path"] = "localization_cues-sal.pickle"
        self._cue_name = params["cue_path"]
        self._data_file = cPickle.load(open(self._cue_name, "rb"))

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].data.shape[0], 20)

    def forward(self, bottom, top):
        top[0].data[...] = 0.0
        for i, image_id in enumerate(bottom[0].data[...]):
            label_i = self._data_file["%i_labels" % image_id]
            top[0].data[i, label_i - 1] = 1.0
            #top[0].data[i, 0] = 1.0  # add background class

class Seg2Tag(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 1:
            raise Exception("The layer need only one inputs!")

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].data.shape[0], 20)

    def forward(self, bottom, top):
        top[0].data[...] = 0.0
        seg_data = np.transpose(bottom[0].data[...], (0, 2, 3, 1))
        batch_size = seg_data.shape[0]
        for k in range(batch_size):
            seg = seg_data[k, :, :, 0]
            tags = np.unique(seg).astype(np.int).tolist()
            for tag in tags:
                if tag == 0 or tag >= 21:
                    continue
                top[0].data[k, tag - 1] = 1.0

class PSLLayer(caffe.Layer):
    def setup(self, bottom, top):
        ## bottom[0]: label, N1HW
        ## bottom[1]: prob, NCHW
        ## top: mask, N1HW
        if len(bottom) != 2:
            raise Exception("PSLLayer need extractly two inputs")
    
    def reshape(self, bottom, top):
        n, _, h, w = bottom[0].data.shape
        top[0].reshape(n, 1, h, w)

    def forward(self, bottom, top):
        # generate online pseudo mask
        batch_size, _, h, w = bottom[0].data.shape
        masks = np.ones(shape=(batch_size, 1, h, w)) * 21
        labels = np.array(bottom[0].data[...]).astype(np.uint8)
        # set_trace()
        probs = np.array(bottom[1].data[...]).astype(np.float)
        probs = probs - np.max(probs, axis=1, keepdims=True)
        assert np.max(probs) < 1
        for k in range(batch_size):
            im_label = np.unique(labels[k])
            prob = probs[k]
            prob[im_label, :, :] += 20
            masks[k, 0, :, :] = np.argmax(prob, axis=0)
        top[0].data[...] = masks
    
    # def backward(self, top, prop_down, bottom):
    #     bottom[1].diff[...] = 0.0
    #     bottom[0].diff[...] = 0.0

# class PSLLayer(caffe.Layer):
#     ## bottom: 0-cues 1-prob(after CRF-log)
#     ## top: 0-argmax(prob[cues]) new seed
#     def setup(self, bottom, top):
#         if len(bottom) != 2:
#             raise Exception("PSLLayer only need two inputs")
#         self._iter = 0
#         params = eval(self.param_str)
#         self._start_iter = params['start_iter']

#     def reshape(self, bottom, top):
#         n, c, h, w = bottom[1].data.shape
#         top[0].reshape(n, 1, h, w)

#     def forward(self, bottom, top):
#         self._iter += 1
#         # if self._iter % 10 == 0:
#         #     print("iter: {}".format(self._iter))
#         if self._iter >= self._start_iter:
#             # set_trace()
#             # generate online seed
#             seeds = np.transpose(np.array(bottom[0].data[...]), [0, 2, 3, 1]).astype(np.uint8)
#             probs = np.transpose(np.array(bottom[1].data[...]), [0, 2, 3, 1]).astype(np.float32)
#             probs = np.exp(probs)
#             results = np.zeros(shape=(probs.shape[0], 1, probs.shape[1], probs.shape[2]))
#             batch_size = seeds.shape[0]
#             for i in range(batch_size):
#                 seed = seeds[i]
#                 # im_labels = np.unique(np.where(seed == 1)[2]).tolist()
#                 im_labels = np.unique(seed).tolist()
#                 if 21 in im_labels:
#                     im_labels.remove(21)
#                 im_non_labels = list(set(range(21)).difference(set(im_labels)))

#                 prob = probs[i]
#                 prob[:, :, im_non_labels] = 0

#                 max_prob = np.max(prob, axis=2)
#                 predict = np.argmax(prob, axis=2)
#                 bg_ratio =  len(np.where(predict == 0)[0]) / predict.size

#                 if bg_ratio > 0.95:
#                     label = np.ones(shape=predict.shape) * 21
#                 else:
#                     label = np.copy(predict)
#                     fg_ignore = np.where((label > 0) & (max_prob < 0.85))
#                     bg_ignore = np.where((label == 0) & (max_prob < 0.99))

#                     label[fg_ignore] = 21
#                     label[bg_ignore] = 21

#                 results[i, 0, :, :] = label

#             top[0].data[...] = results
#         else:
#             n, c, h, w = bottom[1].data.shape
#             top[0].data[...] = np.ones(shape=(n, 1, h, w)) * 21

#     def backward(self, top, prop_down, bottom):
#         bottom[1].diff[...] = 0.0

class PSLLayerAda(caffe.Layer):
    ## bottom: 0-cues 1-prob(after CRF-log)
    ## top: 0-argmax(prob[cues]) new seed
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("PSLLayer only need two inputs")
        self._iter = 0
        params = eval(self.param_str)
        self._start_iter = params['start_iter']
        self._fg_thresh = params['fg_th']
        self._bg_thresh = params['bg_th']

    def reshape(self, bottom, top):
        n, c, h, w = bottom[1].data.shape
        top[0].reshape(n, 1, h, w)

    def forward(self, bottom, top):
        self._iter += 1
        # if self._iter % 10 == 0:
        #     print("iter: {}".format(self._iter))
        if self._iter >= self._start_iter:
            # set_trace()
            # generate online seed
            seeds = np.transpose(np.array(bottom[0].data[...]), [0, 2, 3, 1]).astype(np.uint8)
            probs = np.transpose(np.array(bottom[1].data[...]), [0, 2, 3, 1]).astype(np.float32)
            probs = np.exp(probs)
            results = np.zeros(shape=(probs.shape[0], 1, probs.shape[1], probs.shape[2]))
            batch_size = seeds.shape[0]
            for i in range(batch_size):
                seed = seeds[i]
                im_labels = np.unique(np.where(seed == 1)[2]).tolist()
                im_non_labels = list(set(range(21)).difference(set(im_labels)))

                prob = probs[i]
                prob[:, :, im_non_labels] = 0

                max_prob = np.max(prob, axis=2)
                predict = np.argmax(prob, axis=2)
                bg_ratio =  len(np.where(predict == 0)[0]) / predict.size

                if bg_ratio > 0.95:
                    label = np.ones(shape=predict.shape) * 21
                else:
                    label = np.copy(predict)
                    fg_ignore = np.where((label > 0) & (max_prob < self._fg_thresh))
                    bg_ignore = np.where((label == 0) & (max_prob < self._bg_thresh))

                    label[fg_ignore] = 21
                    label[bg_ignore] = 21

                results[i, 0, :, :] = label

            top[0].data[...] = results
        else:
            n, c, h, w = bottom[1].data.shape
            top[0].data[...] = np.ones(shape=(n, 1, h, w)) * 21

    def backward(self, top, prop_down, bottom):
        bottom[1].diff[...] = 0.0

class AnnotationLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("The layer needs two inputs!")

        layer_params = yaml.load(self.param_str)
        if 'cues' not in layer_params:
            layer_params['cues'] = 'localization_cues.pickle'
        self._cue_name = layer_params['cues']

        if not 'mirror' in layer_params:
            layer_params['mirror'] = False
        self.is_mirror = layer_params['mirror']

        this_dir = osp.dirname(__file__)
        self.data_file = cPickle.load(open(osp.join(this_dir, '../../training', 'localization_cues', self._cue_name)))

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].data.shape[0], 1, 1, 21)
        top[1].reshape(bottom[0].data.shape[0], 21, 41, 41)
        top[2].reshape(*bottom[1].data.shape)

    def forward(self, bottom, top):

        top[0].data[...] = 0.0
        top[1].data[...] = 0.0
        top[2].data[...] = bottom[1].data

        for i, image_id in enumerate(bottom[0].data[...]):

            labels_i = self.data_file['%i_labels' % image_id]
            top[0].data[i, 0, 0, 0] = 1.0
            top[0].data[i, 0, 0, labels_i] = 1.0

            cues_i = self.data_file['%i_cues' % image_id]
            top[1].data[i, cues_i[0], cues_i[1], cues_i[2]] = 1.0

            if self.is_mirror:
                flip = np.random.choice(2) * 2 - 1
                top[1].data[i, ...] = top[1].data[i, :, :, ::flip]
                top[2].data[i, ...] = top[2].data[i, :, :, ::flip]

class AnnotationLayerSoftCue(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("The layer needs two inputs!")

        layer_params = yaml.load(self.param_str)
        if 'cues' not in layer_params:
            layer_params['cues'] = 'localization_cues.pickle'
        self._cue_name = layer_params['cues']

        if not 'mirror' in layer_params:
            layer_params['mirror'] = False
        self.is_mirror = layer_params['mirror']

        this_dir = osp.dirname(__file__)
        self.data_file = cPickle.load(open(osp.join(this_dir, '../../training', 'localization_cues', self._cue_name)))

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].data.shape[0], 1, 1, 21)
        top[1].reshape(bottom[0].data.shape[0], 21, 41, 41)
        top[2].reshape(*bottom[1].data.shape)

    def forward(self, bottom, top):

        top[0].data[...] = 0.0
        top[1].data[...] = 0.0
        top[2].data[...] = bottom[1].data

        for i, image_id in enumerate(bottom[0].data[...]):

            labels_i = self.data_file['%i_labels' % image_id]
            top[0].data[i, 0, 0, 0] = 1.0
            top[0].data[i, 0, 0, labels_i] = 1.0

            cues_i = self.data_file['%i_cues' % image_id]
            top[1].data[i, cues_i[0], cues_i[1], cues_i[2]] = 1.0

            if self.is_mirror:
                flip = np.random.choice(2) * 2 - 1
                top[1].data[i, ...] = top[1].data[i, :, :, ::flip]
                top[2].data[i, ...] = top[2].data[i, :, :, ::flip]

class AnnotationLayerCOCO(caffe.Layer):

    def setup(self, bottom, top):
        layer_params = eval(self.param_str)
        self.source = layer_params['source']
        self.root_folder = layer_params['root']
         # store input as class variables
        self.batch_size = layer_params['batch_size']

        if not 'mirror' in layer_params:
            self.is_mirror = False
        else:
            self.is_mirror = layer_params['mirror']

        self.mean = layer_params['mean']
        self.new_h, self.new_w = layer_params['new_size']
        if not 'ignore_label' in layer_params:
            self.ignore_label = 255
        else:
            self.ignore_label = layer_params['ignore_label']

        self.indexlist = [line.strip().split() for line in open(self.source)]
        self._cur = 0  # current image
        self.q = multiprocessing.Queue(maxsize=self.batch_size*2)
        # self.start_batch()

        top[0].reshape(self.batch_size, 1, 1, 81)
        top[1].reshape(self.batch_size, 81, self.new_h / 8 + 1, self.new_w / 8 + 1)
        top[2].reshape(self.batch_size, 3, self.new_h, self.new_w)

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass

    def forward(self, bottom, top):
        for itt in xrange(self.batch_size):
            # Use the batch loader to load the next image.
            im, label, image_label = self.load_next_image()

            # Add directly to the caffe data layer
            top[0].data[itt, ...] = image_label
            top[1].data[itt, ...] = label
            top[2].data[itt, ...] = im

    def load_next_image(self):
        """
        Load the next image in a batch.
        """
        # Did we finish an epoch?
        if self._cur == len(self.indexlist):
            self._cur = 0
            shuffle(self.indexlist)

        # Load an image
        index = self.indexlist[self._cur]  # Get the image index
        image_file_path, label_file_path = index
        # image = Image.open(osp.join(self.root_folder, image_file_path))
        # label = Image.open(osp.join(self.root_folder, label_file_path))
        image = cv2.imread(self.root_folder+image_file_path, cv2.IMREAD_COLOR)
        label = cv2.imread(self.root_folder+label_file_path, cv2.IMREAD_GRAYSCALE)
        self._cur += 1
        return self.preprocess(image, label)

    def perpare_next_data(self):
        """
        Load the next image in a batch.
        """
        return self.q.get()

    def start_batch(self):
        thread = multiprocessing.Process(target=self.data_generator_task)
        thread.daemon = True
        thread.start()

    def data_generator_task(self):
        while True:
            output = self.load_next_image()
            self.q.put(output)

    def preprocess(self, image, label):
        """
        preprocess() emulate the pre-processing occuring in the vgg16 caffe
        prototxt.
        """
        # image = cv2.convertTo(image, cv2.CV_64F)
        image = np.array(image)
        image = zoom(image.astype('float32'),
                        (self.new_h / float(image.shape[0]),
                        self.new_w / float(image.shape[1]), 1.0),
                        order=1)

        image = image[:, :, [2, 1, 0]]
        image = image - self.mean

        image = image.transpose([2, 0, 1])

        h, w = label.shape
        cues = np.zeros((81, h, w), dtype=np.uint8)
        for (x,y), v in np.ndenumerate(label):
            if not v == self.ignore_label:
                cues[v, x, y] = 1

        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            cues = cues[:, :, ::flip]

        unique_inst = np.unique(label)
        ignore_ind = np.where(unique_inst == self.ignore_label)[0]
        unique_inst = np.delete(unique_inst, ignore_ind)
        image_label = np.zeros((1, 1, 81))
        for cat_id in unique_inst:
            image_label[0, 0, cat_id] = 1

        return image, cues, image_label
