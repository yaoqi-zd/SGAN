from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import pylab, png, os, sys, cv2
import scipy.ndimage as nd
import os.path as osp
from ipdb import set_trace

from pyutils import write_to_png_file, get_part_files

import findcaffe
import krahenbuhl2013

import cPickle, argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_savepath", default=None, type=str)
    parser.add_argument("--png_savepath", default=None, type=str)
    parser.add_argument("--imgPath", default=None, type=str)

    parser.add_argument("--id_path", default=None, type=str)
    
    parser.add_argument("--total_parts", default=1, type=int)
    parser.add_argument("--cur_part", default=1, type=int)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    options = parse_args()
    
    val_ids = open(options.id_path).readlines()
    val_ids = get_part_files(val_ids, options.total_parts, options.cur_part)

    train_flag = True if "train" in options.id_path else False
    if train_flag:
        im_tags = cPickle.load(open("im_tags.pkl", "rb"))

    if not osp.exists(options.png_savepath):
        os.makedirs(options.png_savepath)

    # val_ids = ["2007_005702"]

    # save_path = options.pkl_savepath + "_crf_scale6"
    # if not osp.exists(save_path):
    #     os.mkdir(save_path)

    for _id in val_ids:
        _id = _id.strip()

        image_file = osp.join(options.imgPath, _id + ".jpg")
        im = cv2.imread(image_file)

        probs = cPickle.load(open(osp.join(options.pkl_savepath, _id + ".pkl"), "rb"))

        if train_flag:
            cur_tags = im_tags[_id].tolist()
            cur_tags.append(0)
            non_exist_tags = list(set(range(21)).difference(cur_tags))
            probs[:, :, non_exist_tags] = 0

        eps = 0.00001
        probs[probs < eps] = eps
        # crf_prob = krahenbuhl2013.CRF(im, np.log(probs), scale_factor=6.0)
        # mask = np.argmax(crf_prob, axis=2)
        mask = np.argmax(probs, axis=2)

        # cPickle.dump(crf_prob.astype(np.float16), open(osp.join(save_path, _id + ".pkl"), "wb"), cPickle.HIGHEST_PROTOCOL)

        if train_flag:
            cv2.imwrite(osp.join(options.png_savepath, _id + ".png"), mask)
        else:
            write_to_png_file(mask, osp.join(options.png_savepath, _id + ".png"))
        # write_to_png_file(mask, osp.join(_id + "_crf.png"))
