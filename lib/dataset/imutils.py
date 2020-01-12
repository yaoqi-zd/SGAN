"""
image utils function
"""
import os, math
import os.path as osp
import numpy as np
import cv2
import scipy.ndimage as nd
import random
import torch
import krahenbuhl2013
from multiprocessing import Pool


# random scale
def random_scale_with_ratio(data, label, high_ratio, low_ratio, order=1):
    assert data.ndim == 2 or data.ndim == 3, "only support data with 2d or 3d"
    assert high_ratio >= low_ratio
    
    scale = low_ratio + random.random() * (high_ratio - low_ratio)
    if data.ndim == 3:
        data = nd.zoom(data, (scale, scale, 1.0), order=order)
    else:
        data = nd.zoom(data, (scale, scale), order=order)
    
    label = nd.zoom(label, (scale, scale), order=0)

    return data, label

def random_scale_with_size(data, label, high_size, low_size, order=1):
    assert data.ndim == 2 or data.ndim == 3, "only support data with 2d or 3d"
    assert high_size >= low_size

    size_ = low_size + int(random.random() * (high_size - low_size))
    raw_h, raw_w = data.shape[:2]
    
    if data.ndim == 3:
        data = nd.zoom(data, (size_ / raw_h, size_ / raw_w, 1.0), order=order)
    else:
        data = nd.zoom(data, (size_ / raw_h, size_ / raw_w), order=order)
    
    label = nd.zoom(label, (size_ / raw_h, size_ / raw_w), order=0)

    return data, label

def random_scale_with_size_keep_ratio_list(datas, orders, high_size, low_size, short_thresh):
    """rescale input data sequences with the same aspect ratio, keep short size to short threshold
    
    Parameters
    ----------
    datas : list
        list of data
    orders : list
        list of order
    high_size : int
        max size of long side
    low_size : int
        min size of long side
    short_thresh : int
        min size of short side
    """
    assert high_size >= low_size
    size_ = low_size + int(random.random() * (high_size - low_size))
    raw_h, raw_w = datas[0].shape[:2]

    if raw_h < raw_w:
        short_size, long_size = raw_h, raw_w
    else:
        short_size, long_size = raw_w, raw_h

    ratio = size_ / long_size
    if short_size * ratio < short_thresh:
        ratio = short_thresh / short_size
    
    result = []
    for k, data in enumerate(datas):
        if data.ndim == 3:
            data = nd.zoom(data, (ratio, ratio, 1.0), order=orders[k])
        else:
            data = nd.zoom(data, (ratio, ratio), order=orders[k])
        result.append(data)
    
    return result        

def random_scale_with_size_keep_ratio(data, high_size, low_size, short_thresh, label=None):
    """
    rescale data with the same aspect ratio, keep short size to short threshold
    Parameter
    ---------
    data: image data
    high_size, low_size: random rescale [low_size, high_size]
    short_thresh: threshold of short side after rescale
    label: if label is not None, rescale simutaneously
    """
    assert data.ndim == 2 or data.ndim == 3, "only support data with 2d or 3d"
    assert high_size >= low_size

    size_ = low_size + int(random.random() * (high_size - low_size))
    raw_h, raw_w = data.shape[:2]

    if raw_h < raw_w:
        short_size, long_size = raw_h, raw_w
    else:
        short_size, long_size = raw_w, raw_h
    
    ratio = size_ / long_size
    if short_size * ratio < short_thresh:
        ratio = short_thresh / short_size

    if data.ndim == 3:
        data = nd.zoom(data, (ratio, ratio, 1.0), order=1)
    else:
        data = nd.zoom(data, (ratio, ratio), order=1)

    if label is not None:
        label = nd.zoom(label, (ratio, ratio), order=0)
        return data, label
    return data

def random_crop_list(datas, crop_size, pad_values):
    """random crop input data sequences
    
    Parameters
    ----------
    datas : list    
        list of data to be cropped  
    crop_size : int
        crop size
    pad_values : list
        list of values to be padded
    """
    h, w = datas[0].shape[:2]
    pad_h = max(crop_size - h, 0)
    pad_w = max(crop_size - w, 0)

    assert len(datas) == len(pad_values)

    if pad_h > 0 or pad_w > 0:
        datas_pad = []
        for k in range(len(datas)):
            data_pad = cv2.copyMakeBorder(datas[k], 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_values[k])
            datas_pad.append(data_pad)
    else:
        datas_pad = datas

    new_h, new_w = datas_pad[0].shape[:2]
    h_off = random.randint(0, new_h - crop_size)
    w_off = random.randint(0, new_w - crop_size)

    result = []
    for data_pad in datas_pad:
        if data_pad.ndim == 3:
            data = np.asarray(data_pad[h_off:h_off + crop_size, w_off:w_off + crop_size, :], data_pad.dtype)
        else:
            data = np.asarray(data_pad[h_off:h_off + crop_size, w_off:w_off + crop_size], data.dtype)
        result.append(data)
    return result

def random_crop(data, label, crop_size, data_pad_value, label_pad_value):
    """
    random crop
    if label is not None, crop label simutaneously
    if label is None, return cropped data only
    """
    h, w = data.shape[:2]
    pad_h = max(crop_size - h, 0)
    pad_w = max(crop_size - w, 0)

    if pad_h > 0 or pad_w > 0:
        data_pad = cv2.copyMakeBorder(data, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=data_pad_value)
        if label is not None:
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=label_pad_value)
    else:
        data_pad = data
        if label is not None:
            label_pad = label
    
    new_h, new_w = data_pad.shape[:2]
    h_off = random.randint(0, new_h - crop_size)
    w_off = random.randint(0, new_w - crop_size)

    if data.ndim == 3:
        data = np.asarray(data_pad[h_off:h_off + crop_size, w_off:w_off + crop_size, :], data.dtype)
    else:
        data = np.asarray(data_pad[h_off:h_off + crop_size, w_off:w_off + crop_size], data.dtype)

    if label is not None:
        label = np.asarray(label_pad[h_off:h_off + crop_size, w_off:w_off + crop_size], label.dtype)
        return data, label
    
    return data

def random_jitter(im, bright_value=32, hue_value=18, contrast_value=(0.5, 1.5), saturate_value=(0.5, 1.5)):

    def random_brightness(im, prob, delta):
        """Do random brightness distortion"""
        assert 0 <= prob <= 1
        assert delta > 0
        rng = random.random()
        if rng < prob:
            rng_delta = random.uniform(-1 * delta, 1 * delta)
            out_im = cv2.convertScaleAbs(im, alpha=1, beta=rng_delta)
            return out_im
        else:
            return im

    def random_contrast(im, prob, lower, upper):
        """Do random contrast distortion"""
        assert 0 <= prob <= 1
        assert upper > lower
        assert lower > 0

        rng = random.random()
        if rng < prob:
            rng_delta = random.uniform(lower, upper)
            out_im = cv2.convertScaleAbs(im, alpha=rng_delta, beta=0)
            return out_im
        else:
            return im

    def random_saturation(im, prob, lower, upper):
        """ Do random saturation distortion"""
        assert 0 <= prob <= 1
        assert upper > lower
        assert lower > 0

        rng = random.random()
        if rng < prob:
            rng_delta = random.uniform(lower, upper)
            if math.fabs(rng_delta - 1.0) != 0.001:
                hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
                hsv_im[:, :, 1] = cv2.convertScaleAbs(hsv_im[:, :, 1], alpha=rng_delta, beta=0)
                out_im = cv2.cvtColor(hsv_im, cv2.COLOR_HSV2BGR)
                return out_im
            else:
                return im
        else:
            return im

    def random_hue(im, prob, delta):
        """ Do random hue distortion"""
        assert delta > 0
        rng = random.random()
        if rng < prob:
            rng_delta = random.uniform(-1 * delta, delta)
            hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            hsv_im[:, :, 0] = cv2.convertScaleAbs(hsv_im[:, :, 0], alpha=1, beta=rng_delta)
            out_im = cv2.cvtColor(hsv_im, cv2.COLOR_HSV2BGR)
            return out_im
        else:
            return im

    im = random_brightness(im, 0.5, bright_value)
    im = random_contrast(im, 0.5, contrast_value[0], contrast_value[1])
    im = random_saturation(im, 0.5, saturate_value[0], saturate_value[1])
    im = random_hue(im, 0.5, hue_value)

    return im


def generate_att_mask_from_seed(seed):
    """
    generate balance mask for self-attention from seed
    support seed can be divide into background|ignore|foreground region with pixel
    numbers a, b, c respectively.
    then ratios of background|ignore|foreground are computed as:

    ratio_back = 1 / (1 + a / b + a / c)
    ratio_ignore = ratio_back * (a / b)
    ratio_fore = ratio_back * (a / c)
    """
    att_mask = np.zeros(seed.shape, dtype=np.float)
    zero_ind = np.where(seed == 0)
    ignore_ind = np.where(seed == 21)
    foreground_ind = np.where((seed > 0) & (seed < 21))
    len_zero, len_ignore, len_fore = len(zero_ind[0]) + 20, len(ignore_ind[0]) + 20, len(foreground_ind[0]) + 20
    ratio_zero = 1 / (1 + len_zero / len_ignore + len_zero / len_fore)
    ratio_ignore = ratio_zero * len_zero / len_ignore
    ratio_fore = ratio_zero * len_zero / len_fore
    att_mask[zero_ind] = ratio_zero
    att_mask[ignore_ind] = ratio_ignore
    att_mask[foreground_ind] = ratio_fore

    return att_mask


def denormalizing(ori_images, mean=(0, 0, 0), std=None):
    """
    denormalizing tensor images with mean and std
    Args:
        images: tensor, N*C*H*W
        mean: tuple
        std: tuple
    """

    images = torch.tensor(ori_images, requires_grad=False)

    if std is not None:
        """pytorch style normalization"""
        mean = torch.tensor(mean).view((1, 3, 1, 1))
        std = torch.tensor(std).view((1, 3, 1, 1))

        images *= std
        images += mean
        images *= 255

        images = torch.flip(images, dims=(1,))
    else:
        """caffe style normalization"""
        mean = torch.tensor(mean).view((1, 3, 1, 1))
        images += mean

    return images

def crf_single(inp):
    im, prob = inp[0], inp[1]
    im = np.squeeze(im, axis=0)
    prob = np.squeeze(prob, axis=0)
    return krahenbuhl2013.CRF(im, prob, scale_factor=12.0)

def crf_with_alpha(img, cams, alpha):
    """
    crf inference with addional background class

    Parameter
    ---------
    img: [N, h, w, 3]
    cams: [N, h, w, C]
    alpha: param to control background prob

    Return
    ------
    crf_prob: [N, h, w, C+1]
    """

    batch_size = img.shape[0]

    bg_score = np.power(1 - np.max(cams, axis=3, keepdims=True), alpha)
    bg_cam_score = np.concatenate((bg_score, cams), axis=3)

    list_ims = np.split(img, batch_size, axis=0)
    list_probs = np.split(bg_cam_score, batch_size, axis=0)
    list_input = zip(list_ims, list_probs)

    pool = Pool(processes=6)
    list_crf = pool.map(crf_single, list_input)
    pool.close()
    pool.join()
    crf_prob = np.stack(list_crf, axis=0)

    return crf_prob


def generate_seed(ims, cams, alpha_range=[], crf_scale=10, class_num=21):
    """
    generate seed given cam and images

    Parameter
    ---------
    ims: [N, 3, H, W]
    cams: [N, C, h, w]
    alpha_range: [low_alpha, high_alpha]
    crf_scale: scale param for crf inference
    class_num: class numbers of the dataset

    Return
    ------
    label: [N, h, w]
    """
    ims = np.transpose(ims, axes=(0, 2, 3, 1))
    cams = np.transpose(cams, axes=(0, 2, 3, 1))

    _, h, w, _ = cams.shape
    _, H, W, _ = ims.shape
    
    ims = nd.zoom(ims, (1, h / H, w / W, 1), order=1)

    assert len(alpha_range) == 2
    assert alpha_range[0] < alpha_range[1]

    low_alpha_crf = crf_with_alpha(ims, cams, alpha_range[0])
    high_alpha_crf = crf_with_alpha(ims, cams, alpha_range[1])


    joint_crf = np.concatenate((low_alpha_crf, high_alpha_crf), axis=3)
    no_score_region = np.max(joint_crf, axis=-1) < 1e-5

    label_la = np.argmax(low_alpha_crf, axis=-1).astype(np.uint8)
    label_ha = np.argmax(high_alpha_crf, axis=-1).astype(np.uint8)

    label = label_la.copy()
    label[label_la == 0] = class_num
    label[label_ha == 0] = 0
    label[no_score_region] = class_num

    return label
