
import numpy as np
import time
import sys
from ipdb import set_trace
import yaml
from munch import munchify
from matplotlib import colors as mpl_colors
import png

palette = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.5, 0.0),
           (0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5), (0.5, 0.5, 0.5),
           (0.25, 0.0, 0.0), (0.75, 0.0, 0.0), (0.25, 0.5, 0.0), (0.75, 0.5, 0.0),
           (0.25, 0.0, 0.5), (0.75, 0.0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5),
           (0.0, 0.25, 0.0), (0.5, 0.25, 0.0), (0.0, 0.75, 0.0), (0.5, 0.75, 0.0),
           (0.0, 0.25, 0.5)]
my_cmap = mpl_colors.LinearSegmentedColormap.from_list('Custom cmap', palette, 21)

def write_to_png_file(im, f):
    global palette
    palette_int = map(lambda x: map(lambda xx: int(255 * xx), x), palette)
    w = png.Writer(size=(im.shape[1], im.shape[0]), bitdepth=8, palette=palette_int)
    with open(f, "w") as ff:
        w.write(ff, im)

def read_yaml2cls(yml_file):
    """read yml file and convert to class"""
    with open(yml_file, "r") as stream:
        data = yaml.safe_load(stream)
    cls_data = munchify(data)
    return cls_data

def get_part_files(filelists, total_parts, cur_part):
    assert cur_part <= total_parts
    assert cur_part >= 1

    part_len = int(len(filelists) / total_parts)
    start_ind = part_len * (cur_part - 1)
    end_ind = part_len * cur_part if cur_part != total_parts else len(filelists)
    return filelists[start_ind : end_ind]

class Logger(object):
    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "w")
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()


class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v


class Timer:
    def __init__(self, starting_msg = None):
        self.start = time.time()
        self.stage_start = self.start

        if starting_msg is not None:
            print(starting_msg, time.ctime(time.time()))


    def update_progress(self, progress):
        self.elapsed = time.time() - self.start
        self.est_total = self.elapsed / progress
        self.est_remaining = self.est_total - self.elapsed
        self.est_finish = int(self.start + self.est_total)


    def str_est_finish(self):
        return str(time.ctime(self.est_finish))

    def get_stage_elapsed(self):
        return time.time() - self.stage_start

    def reset_stage(self):
        self.stage_start = time.time()


class Evaluator(object):
    def __init__(self, num_class, ignore=False):
        self.num_class = num_class
        self.ignore = ignore  # whether to consider ignore class, True when evaluate seed quality
        self.confusion_matrix = np.zeros(shape=(self.num_class, self.num_class))  # index 0: gt / index 1: pred

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Precision_Recall(self):
        precision = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=0) + 1e-5)
        recall = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) + 1e-5)
        if self.ignore:
            mp = np.nanmean(precision[:-1])
            mr = np.nanmean(recall[:-1])
            return precision[:-1], recall[:-1], mp, mr
        else:
            mp = np.nanmean(precision)
            mr = np.nanmean(recall)
            return precision, recall, mp, mr

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        if self.ignore:
            MIoU = np.nanmean(IoU[:-1])
            return IoU[:-1], MIoU
        else:
            MIoU = np.nanmean(IoU)
            return IoU, MIoU

    def Mean_Intersection_over_Union_ignore(self):
        """
        compute mIoU when ignore region do not consider, used when estimate label quality
        """
        if self.ignore:
            confusion_matrix = self.confusion_matrix[:-1, :-1]
        IoU = np.diag(confusion_matrix) / (
                    np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                    np.diag(confusion_matrix))
        MIoU = np.nanmean(IoU)
        return IoU, MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape, "gt: {} pred: {}".format(gt_image.shape, pre_image.shape)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)