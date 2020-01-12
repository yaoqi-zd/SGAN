
import numpy as np
import time
import sys
from ipdb import set_trace
import yaml
from munch import munchify
import krahenbuhl2013
import scipy.ndimage as nd
from PIL import Image


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


def softmax(scores, axis=-1):
    # normalize
    max_scores = np.max(scores, axis=axis, keepdims=True)
    scores -= max_scores
    scores = np.exp(scores)
    probs = scores / np.sum(scores, axis=axis, keepdims=True)

    return probs    

def normalize(scores):
    """
    normalize in spatial dimension

    Parameter
    ---------
    scores: [H, W, C]

    return
    ------
    norm_scores: [H, W, C]
    """
    h, w, c = scores.shape
    score_reshape = np.reshape(scores, newshape=(-1, c))
    min_score = np.min(score_reshape, axis=0, keepdims=True)
    max_score = np.max(score_reshape, axis=0, keepdims=True)
    norm_score_reshape = (score_reshape - min_score) / (max_score - min_score + 1e-5)
    norm_score = np.reshape(norm_score_reshape, newshape=(h, w, c))

    return norm_score

def entropy(scores, norm=True):
    """
    compute per-localization entropy map for input score map
    
    Parameters
    ----------
    scores: (H, W, C)

    Return
    ------
    entropy: (H, W)
    """
    if norm:
        probs = softmax(scores)
    else:
        probs = scores.copy()

    # compute entropy
    probs[probs < 1e-5] = 1e-5
    entropy = np.sum(-1 * probs * np.log2(probs), axis=-1)

    return entropy

def CRF(im, score, scale=12.0):
    """
    perform CRF process
    
    Parameter
    ---------
    im: [H, W, 3]
    score: [h, w, c]
    scale: scale factor of CRF
    """
    im_h, im_w = im.shape[:2]
    h, w = score.shape[:2]
    if im_h != h:
        im = nd.zoom(im, (h / im_h, w / im_w, 1.0), order=1)
    
    prob = softmax(score)
    return krahenbuhl2013.CRF(im, prob, scale_factor=scale)


def crf_inference(img, probs, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))

def mask2png(saven, mask):
    palette = get_palette(256)
    mask=Image.fromarray(mask.astype(np.uint8))
    mask.putpalette(palette)
    mask.save(saven)

def get_palette(num_cls):
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def generate_seed_wo_ignore(localization, train_boat=True):
    """
    This function generate seed with priority strategy
    :param localization:
    :return:
    """
    h, w, c = localization.shape
    assert c == 21

    # generate background seed
    mask = np.ones((h, w), dtype=np.int) * 21
    bg_ind = np.where(localization[:, :, 0])
    mask[bg_ind[0], bg_ind[1]] = 0

    # generate foreground seed in the order of their area
    area = np.sum(localization, axis=(0, 1))
    cls_order = np.argsort(area)[::-1]  # area in descending order
    for cls in cls_order:
        if area[cls] == 0:
            break
        ind = np.where(localization[:, :, cls])
        mask[ind[0], ind[1]] = cls

    if train_boat:
        train_boat_ind = np.where(((mask == 4) | (mask == 19)) & (localization[:, :, 0] == 1))
        mask[train_boat_ind] = 0

    return mask

def get_localization_cues_sec(att_maps, saliency, im_label, cam_thresh):
    """get localization cues with method in SEC paper
    perform hard thresholding for each foreground class

    Parameters
    ----------
    att_maps: [h, w, 20]
    saliency: [H, W] normalized saliency maps (0~1)
    im_label: list of foreground classes, aero:1
    cam_thresh: hard threshold to extract foreground class cues

    Return
    ------
    seg_mask: [h, w]
    """
    h, w = att_maps.shape[:2]
    im_h, im_w = saliency.shape[:2]

    localization1 = np.zeros(shape=(h, w, 21))
    for idx in im_label:  # idx: aero=1
        heat_map = att_maps[:, :, idx - 1]
        localization1[:, :, idx] = heat_map > cam_thresh * np.max(heat_map)

    if im_h != h:
        saliency = nd.zoom(saliency, (h / im_h, h / im_w), order=1)
    localization1[:, :, 0] = saliency < 0.06

    # handle conflict seed
    seg_mask = generate_seed_wo_ignore(localization1, train_boat=True)

    return seg_mask

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


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.num = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.num += n
        self.avg = self.sum / self.num


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