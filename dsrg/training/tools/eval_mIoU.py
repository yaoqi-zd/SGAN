import os
import os.path as osp
import numpy as np
import cv2, argparse
from PIL import Image
import pylab
from pyutils import Evaluator, read_yaml2cls
from ipdb import set_trace
import shutil
import time

classes=['background', 'aeroplane', 'bicycle', 'bird','boat', 'bottle', 'bus', 'car', 'cat', 
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 
        'sheep', 'sofa', 'train', 'tvmonitor']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_path", default=None, type=str)
    parser.add_argument("--num_classes", default=21, type=int)
    args = parser.parse_args()

    return args 


if __name__ == '__main__':
    args = parse_args()
    
    res_path = args.res_path
    # DO NOT USE SEGMENTATIONCLASSAUG dir for segmentation evaluation, the result is not correct! 
    root_path = "/data/yaoqi/Dataset/VOCdevkit/VOC2012/"
    gt_path = osp.join(root_path, "SegmentationClassAug") if args.num_classes > 21 else osp.join(root_path, "SegmentationClass")

    names = os.listdir(res_path)
    ignore = True if args.num_classes > 21 else False
    evaluator = Evaluator(num_class=args.num_classes, ignore=ignore)
    for k, name in enumerate(names):
        res = np.array(Image.open(osp.join(res_path, name)))
        #assert osp.exists(osp.join(res_path, name))
        #res = cv2.imread(osp.join(res_path, name), cv2.IMREAD_GRAYSCALE)
        #print(res.shape)
        gt = np.array(Image.open(osp.join(gt_path, name)))

        evaluator.add_batch(gt, res)

        if (k + 1) % 1000 == 0:
            print("evaluation: finish {} files".format(k + 1))

    IoU, mIoU = evaluator.Mean_Intersection_over_Union()
    pre, rec, mp, mr = evaluator.Precision_Recall()
    if ignore:
        # print seed quality
        print("{:<15s}\t{:<15s}\t{:<15s}".format("class", "precision", "recall"))
        str_format = "{:<15s}\t{:<15.2%}\t{:<15.2%}"
        for k in range(21):
            print(str_format.format(classes[k], pre[k], rec[k]))
        print("mIoU = {:.2%} mp = {:.2%} mr = {:.2%}".format(mIoU, mp, mr))
    else:
        # print segmentation IoU
        print("{:<15s}\t{:<15s}".format("class", "IoU"))
        str_format = "{:<15s}\t{:<15.2%}"
        for k in range(21):
            print(str_format.format(classes[k], IoU[k]))
        print("mIoU = {:.2%}".format(mIoU))

        with open("eval_result.txt", "w") as f:
            res_str = "&%.1f  " * 22
            res_str = res_str.strip()
            data = IoU.tolist()
            data.insert(0, mIoU)
            data = [100 * item for item in data]
            f.write(res_str % tuple(data))