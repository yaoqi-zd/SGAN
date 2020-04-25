from __future__ import print_function

from __future__ import division
from __future__ import absolute_import

import os
import os.path as osp
import numpy as np
import cv2
import matplotlib.colors as mpl_colors
import png
from multiprocessing import Pool

palette = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.5, 0.0),
           (0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5), (0.5, 0.5, 0.5),
           (0.25, 0.0, 0.0), (0.75, 0.0, 0.0), (0.25, 0.5, 0.0), (0.75, 0.5, 0.0),
           (0.25, 0.0, 0.5), (0.75, 0.0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5),
           (0.0, 0.25, 0.0), (0.5, 0.25, 0.0), (0.0, 0.75, 0.0), (0.5, 0.75, 0.0),
           (0.0, 0.25, 0.5), (0.75, 0.75, 0.75)]
my_cmap = mpl_colors.LinearSegmentedColormap.from_list('Custom cmap', palette, 22)

def write_to_png_file(im, f):
    global palette
    palette_int = map(lambda x: map(lambda xx: int(255 * xx), x), palette)
    w = png.Writer(size=(im.shape[1], im.shape[0]), bitdepth=8, palette=palette_int)
    with open(f, "w") as ff:
        w.write(ff, im)

#single_channel_path = "/data1/yaoqi/segmentation/weakly/SSENet/result/conf_seed_tb_8.0_24.0"
single_channel_path = "/data/yaoqi/segmentation/wsss/sgan/result/collab/collab_vgg16_321x321_cls_bg_label"

#single_channel_path = "/data/yaoqi/Dataset/VOCdevkit/VOC2012/ini_seed/semi_30_bg_500"

color_path = single_channel_path + "_color"
if not osp.exists(color_path):
    os.mkdir(color_path)

names = os.listdir(single_channel_path)

def single2color(name):
    im = cv2.imread(osp.join(single_channel_path, name), 0)
    write_to_png_file(im, osp.join(color_path, name))

pool = Pool(processes=4)
pool.map(single2color, names)
pool.close()
pool.join()

