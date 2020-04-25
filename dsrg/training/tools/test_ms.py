#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-10-15 下午7:52
# @Author  : yaoqi
# @Email   : yaoqi_isee@zju.edu.cn 
# @File    : test_deeplabv2_weakly_ms.py

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import pylab, png, os, sys, cv2
import scipy.ndimage as nd
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors
from scipy.io import savemat
from ipdb import set_trace
from pyutils import write_to_png_file, get_part_files

import findcaffe
import krahenbuhl2013
import caffe
from optparse import OptionParser
import cPickle

parser = OptionParser()
# parser.add_option("--image", dest="image_file", type='string',
#                  help="Path to image")
parser.add_option("--model", dest="model", type='string',
                  help="Model weights")
parser.add_option("--net", dest="net", type='string', help="Caffe Net")
parser.add_option("--imgPath", dest='imgPath', type='string', help='Image base path')
parser.add_option("--png_savepath", dest="png_savepath", type='string',
                  help="Output png save path", default='')
parser.add_option('--mat_savepath', dest='mat_savepath', type='string',
                  help='Output mat save path', default='')
parser.add_option('--id_path', dest='id_path', type='string',
                  help='File for ids')
parser.add_option("--smooth", dest="smooth", action='store_true', default=False,
                  help="Apply postprocessing")
parser.add_option("--multi_scale", dest="multi_scale", action="store_true", default=False)

parser.add_option("--gpu", dest="gpu", type='int', help="GPU device")
parser.add_option('--out_blob', dest='out_blob', type='string', default='fc8_merge',
                  help='output blob name')
parser.add_option("--total_parts", default=1, type=int)
parser.add_option("--cur_part", default=1, type=int)

options, _ = parser.parse_args()

caffe.set_device(options.gpu)
caffe.set_mode_gpu()

mean_pixel = np.array([104.0, 117.0, 123.0])

# palette = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.5, 0.0),
#            (0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5), (0.5, 0.5, 0.5),
#            (0.25, 0.0, 0.0), (0.75, 0.0, 0.0), (0.25, 0.5, 0.0), (0.75, 0.5, 0.0),
#            (0.25, 0.0, 0.5), (0.75, 0.0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5),
#            (0.0, 0.25, 0.0), (0.5, 0.25, 0.0), (0.0, 0.75, 0.0), (0.5, 0.75, 0.0),
#            (0.0, 0.25, 0.5)]
# my_cmap = mpl_colors.LinearSegmentedColormap.from_list('Custom cmap', palette, 21)


# def write_to_png_file(im, f):
#     global palette
#     palette_int = map(lambda x: map(lambda xx: int(255 * xx), x), palette)
#     w = png.Writer(size=(im.shape[1], im.shape[0]), bitdepth=8, palette=palette_int)
#     with open(f, "w") as ff:
#         w.write(ff, im)

def preprocess(image, size, mean_pixel=mean_pixel):
    image = np.array(image)
    image = nd.zoom(image.astype('float32'),
                    (size / float(image.shape[0]),
                    size / float(image.shape[1]), 1.0),
                    order=1)

    image = image[:, :, [2, 1, 0]]
    image = image - mean_pixel

    image = image.transpose([2, 0, 1])
    return np.expand_dims(image, 0)


def predict_mask(image_file, net, smooth=True, multi_scale=True):

    im = pylab.imread(image_file)
    d1, d2 = float(im.shape[0]), float(im.shape[1])

    scores_all = 0
    scales = [321] if multi_scale is False else [int(0.75*385), 385, int(1.25*385)]
    for im_size in scales:

        im_process = preprocess(im, im_size)
        net.blobs['images'].reshape(*im_process.shape)
        net.blobs['images'].data[...] = im_process

        net.forward()
        scores = np.transpose(net.blobs[options.out_blob].data[0], [1, 2, 0])
        scores = nd.zoom(scores, (d1 / scores.shape[0], d2 / scores.shape[1], 1.0), order=1)

        # set_trace()

        # im_flip = im_process[:, :, :, ::-1]
        # net.blobs['images'].reshape(*im_flip.shape)
        # net.blobs['images'].data[...] = im_flip
        # net.forward()
        # score_flip = np.transpose(net.blobs[options.out_blob].data[0], [1, 2, 0])
        # score_flip = nd.zoom(score_flip, (d1 / score_flip.shape[0], d2 / score_flip.shape[1], 1.0), order=1)
        # score_flip = score_flip[:, ::-1, :]

        scores_all += scores
        # scores_all += score_flip

    scores_exp = np.exp(scores_all - np.max(scores_all, axis=2, keepdims=True))
    probs = scores_exp / np.sum(scores_exp, axis=2, keepdims=True)

    eps = 0.00001
    probs[probs < eps] = eps

    if smooth:
        result = krahenbuhl2013.CRF(im, np.log(probs), scale_factor=1.0)
    else:
        result = probs

    return result

if __name__ == "__main__":

    model = options.model
    net = caffe.Net(options.net, model, caffe.TEST)

    # add by Baisheng
    image_source_path = options.imgPath
    png_save_path = options.png_savepath

    if not os.path.exists(png_save_path):
        os.mkdir(png_save_path)

    val_ids = open(options.id_path).readlines()
    val_ids = get_part_files(val_ids, options.total_parts, options.cur_part)

    train_flag = True if "train" in options.id_path else False
    if train_flag:
        im_tags = cPickle.load(open("im_tags.pkl", "rb"))

    for _id in val_ids:
        _id = _id.strip()
        output = os.path.join(png_save_path, _id + '.png')
        if os.path.exists(output):
            continue
        image_file = os.path.join(image_source_path, _id + '.jpg')
        if options.smooth:
            prob = predict_mask(image_file, net, smooth=options.smooth, multi_scale=options.multi_scale)

            if train_flag:
                cur_tags = im_tags[_id].tolist()
                cur_tags.append(0)
                prob[:, :, cur_tags] += 1
    
            label = np.argmax(prob, axis=2)
            if train_flag:
                cv2.imwrite(output, label)
            else:
                write_to_png_file(label, output)
        else:
            prob = predict_mask(image_file, net, smooth=False, multi_scale=options.multi_scale)
            pkl_save_path = os.path.join("/data/yaoqi/segmentation/dsrg/" , png_save_path)
            if not os.path.exists(pkl_save_path):
                os.makedirs(pkl_save_path)
            output = os.path.join(pkl_save_path, _id + ".pkl")
            cPickle.dump(prob.astype(np.float16), open(output, "wb"), cPickle.HIGHEST_PROTOCOL)


