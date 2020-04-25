import findcaffe
import caffe

import numpy as np
import sys
import os
import os.path as osp

data_path = "/data1/yaoqi/segmentation/weakly/wsss/data/pretrained/"
# prototxt = osp.join(data_path, "vgg16_20M.prototxt")
prototxt = "training/experiment/anti-noise/config/deeplabv2_weakly.prototxt"
pretrain = osp.join(data_path, "vgg16_20M_mc.caffemodel")

net = caffe.Net(prototxt, caffe.TEST)
net.copy_from(pretrain)

for layer_name, param in net.params.iteritems():
    weight = param[0].data[...]
    print("{:<15}: max: {:.4f} min: {:.4f} mean: {:.4f} shape: {}".format(layer_name,
                                                                          np.max(weight),
                                                                          np.min(weight),
                                                                          np.mean(weight),
                                                                          weight.shape))
    if len(param) > 1:
        bias = param[1].data[...]
        print("{:<15}: max: {:.4f} min: {:.4f} mean: {:.4f} shape: {}".format(layer_name,
                                                                          np.max(bias),
                                                                          np.min(bias),
                                                                          np.mean(bias),
                                                                          bias.shape))                                                                          
