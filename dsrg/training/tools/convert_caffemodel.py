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


if __name__ == "__main__":
    config_name = "config/deeplabv2_weakly.prototxt"
    model_name = "model/seed_collab/model_iter_8000.caffemodel"
    net = caffe.Net(config_name, model_name, caffe.TEST)

    data = dict()
    for layer_name in net.params.keys():
        if "fc8" in layer_name:
            save_layer_name = "fc8" + "_" + layer_name.split("_")[-1]
        else:
            save_layer_name = layer_name
        
        weight = net.params[layer_name][0].data[...].copy()
        data[save_layer_name + ".weight"] = weight
        print("{:<10s}: mean: {:.4f}".format(save_layer_name + ".weight", np.mean(weight)))

        if len(net.params[layer_name]) > 1:
            bias = net.params[layer_name][1].data[...].copy()
            data[save_layer_name + ".bias"] = bias
            print("{:<10s}: mean: {:.4f}".format(save_layer_name + ".bias", np.mean(bias)))

    cPickle.dump(data, open("seed_collab_caffe_8k.pth", "wb"), cPickle.HIGHEST_PROTOCOL)
