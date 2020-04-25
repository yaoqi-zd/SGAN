import os
import os.path as osp
import numpy as np
import cv2, argparse
from PIL import Image
import pylab, pickle
from pyutils import Evaluator, read_yaml2cls
import shutil
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_path", default=None, type=str)
    parser.add_argument("--num_classes", default=21, type=int)
    parser.add_argument("--save_path", default=None, type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    args.res_path = "/data1/yaoqi/segmentation/weakly/wsss/dsrg/training/experiment/anti-noise/result/seg_res_trainaug_5755/"
    args.save_path = "seed_mIoU"
    args.num_classes = 22

    res_path = args.res_path
    if not osp.exists(args.save_path):
        os.makedirs(args.save_path)

    gt_path = "/data1/yaoqi/Dataset/VOCdevkit/VOC2012/SegmentationClassAug/"
    gt_rgb_path = "/data1/yaoqi/Dataset/VOCdevkit/VOC2012/SegmentationClassAug_color/"

    names = os.listdir(res_path)
    ignore = True if args.num_classes > 21 else False
    evaluator = Evaluator(num_class=args.num_classes, ignore=ignore)
    # names = ["2008_007519.png"]

    name_mIoU = dict()
    for k, name in enumerate(names):
        res = np.array(Image.open(osp.join(res_path, name)))
        gt = np.array(Image.open(osp.join(gt_path, name)))
        gt_rgb = cv2.imread(osp.join(gt_rgb_path, name))

        evaluator.add_batch(gt, res)

        _, mIoU = evaluator.Mean_Intersection_over_Union_ignore()
        
        f = plt.figure(facecolor="white")
        ax = f.add_subplot(1, 2, 1)
        res[0, 0] = 21
        ax.imshow(res, cmap=my_cmap)
        ax.axis("off")

        ax = f.add_subplot(1, 2, 2)
        ax.imshow(gt_rgb[:, :, ::-1])
        ax.axis("off")

        new_name = "{:.4f}_{}".format(mIoU, name)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        plt.savefig(os.path.join(args.save_path, new_name))
        plt.close()
        
        evaluator.reset()
        name_mIoU[name.split(".png")[0]] = mIoU
        if (k + 1) % 500 == 0:
            print("finish {} files".format(k + 1))
    
    pickle.dump(name_mIoU, open("name_mIoU_5755.pkl", "wb"), pickle.HIGHEST_PROTOCOL)

