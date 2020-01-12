import os, pickle
import os.path as osp
import numpy as np
import cv2
import scipy.ndimage as nd
import init_path
from lib.dataset.get_dataset import get_dataset
from lib.network.sgan import SGAN

import torch
from torch.utils.data import DataLoader
import argparse
from ipdb import set_trace
import matplotlib.pyplot as plt
from lib.utils import pyutils

classes=['background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", default=None, type=str)
    args = parser.parse_args()
    args = pyutils.read_yaml2cls(args.cfg_file)

    return args

# mean pixel : in B-G-R channel order
mean_pixel = np.array([104.008, 116.669, 122.675])

def preprocess(image, size):
    """ pre-process images with Opencv format"""
    image = np.array(image)
    H, W, _ = image.shape
    image = nd.zoom(image.astype('float32'), (size / H, size / W, 1.0), order=1)

    image = image - mean_pixel
    image = image.transpose([2, 0, 1])
    image = np.expand_dims(image, axis=0)

    return torch.from_numpy(image)


def generate_seed_with_ignore(localization):
    """
    This function generate seed ignoring all the conflicts
    :param localization: (41, 41, 21) binary value
    :return:
    """
    h, w, c = localization.shape
    assert (h == 41) & (w == 41) & (c == 21)

    # set_trace()

    # find conflict index
    sum_loc = np.sum(localization, axis=2)
    conflict_ind = np.where(sum_loc > 1)

    # set conflict position to 0
    localization[conflict_ind[0], conflict_ind[1], :] = 0

    # generate seed
    ind = np.where(localization)
    mask = np.ones(shape=(h, w), dtype=np.int) * 21
    mask[ind[0], ind[1]] = ind[2]

    return mask

def generate_seed_wo_ignore(localization, train_boat=False):
    """
    This function generate seed with priority strategy
    :param localization:
    :return:
    """
    h, w, c = localization.shape
    assert (h == 41) & (w == 41) & (c == 21)

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
    att_maps: [41, 41, 20]
    saliency: [H, W]
    im_label: list of foreground classes
    cam_thresh: hard threshold to extract foreground class cues

    Return
    ------
    seg_mask: [41, 41]
    """
    h, w = att_maps.shape[:2]
    im_h, im_w = saliency.shape[:2]

    localization1 = np.zeros(shape=(h, w, 21))
    for idx in im_label:  # idx: aero=1
        heat_map = att_maps[:, :, idx - 1]
        localization1[:, :, idx] = heat_map > cam_thresh * np.max(heat_map)

    # bg_cue = saliency.astype(np.float32)
    # bg_cue = bg_cue / 255

    bg_cue = nd.zoom(saliency, (h / im_h, h / im_w), order=1)
    localization1[:, :, 0] = bg_cue < 0.06

    # handle conflict seed
    if args.ignore_conflict:
        seg_mask = generate_seed_with_ignore(localization1)
    else:
        seg_mask = generate_seed_wo_ignore(localization1, train_boat=True)

    return seg_mask

def get_localization_cues_dcsp(att_maps, saliency, im_label, bg_thresh):
    """get localization cues with method in DCSP paper
    compute harmonic mean for each foreground class

    Parameters
    ----------
    att_maps: [41, 41, 20]
    saliency: [H, W]
    im_label: list of foreground classes
    cam_thresh: hard threshold to extract foreground class cues

    Return
    ------
    seg_mask: [41, 41]
    """
    h, w = att_maps.shape[:2]
    im_h, im_w = saliency.shape[:2]

    re_sal = nd.zoom(saliency, (h / im_h, w / im_w), order=1)
    localization1 = np.zeros(shape=(h, w, 20))
    for idx in im_label: # idx: aero=1
        localization1[:, :, idx - 1] = 2 / ((1 / (att_maps[:, :, idx - 1] + 1e-7)) + (1 / (re_sal + 1e-7)))
    hm_max = np.max(localization1, axis=2)
    seg_mask = np.argmax(localization1, axis=2) + 1
    seg_mask[hm_max < bg_thresh] = 0

    return seg_mask

def filter_weight_dict(weight_dict, model_dict):
    # filter the parameters that exist in the pretrained model
    pretrained_dict = dict()
    for k, v in weight_dict.items():
        # keep compatable with the previous version of network definition
        if "conv" in k and "backbone" not in k:
            k = "backbone." + k
        if k in model_dict:
            pretrained_dict[k] = v
    
    model_dict.update(pretrained_dict)
    return model_dict


if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda:0")

    # input and output
    im_tags = pickle.load(open(args.cue_file, "rb"))
    if not osp.exists(args.res_path):
        os.mkdir(args.res_path)
    
    _, test_dataset = get_dataset(args.dataset_name, args)

    batch_size = 8
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # load net and trained weights
    model = SGAN(backbone_name=args.backbone)
    weight_dict = torch.load(osp.join(args.save_model_path, args.cfg_name, "model_iter_" + str(args.max_iter) + ".pth"))
    model_dict = filter_weight_dict(weight_dict, model.state_dict())
    model.load_state_dict(model_dict)

    model = model.to(device)
    model.eval()

    save_path = osp.join(args.res_path, args.cfg_name + args.test_cfg)
    if not osp.exists(save_path):
        os.makedirs(save_path)

    # compute class activation map
    with torch.no_grad():
        for num, pack in enumerate(test_loader):
            names, imgs, labels = pack[0], pack[1].to(device, dtype=torch.float32), \
                                          pack[2].numpy()
            
            fg_sim = pack[3].to(device, dtype=torch.float32)
            bg_sim = pack[4].to(device, dtype=torch.float32)
            sizes = pack[6].to("cpu").numpy()
            if args.combine_seedseg:
                _, segs, cams = model.forward_cam(imgs, fg_sim, bg_sim)
                cams = cams + segs
                # cams = segs
            else:
                _, _, cams = model.forward_cam(imgs, fg_sim, bg_sim)

            np_cams = np.transpose(cams.cpu().numpy(), (0, 2, 3, 1))
            _, h, w, c = np_cams.shape

            for k, name in enumerate(names):

                # get output cam
                im_label = im_tags[name]
                im_h, im_w = sizes[k]
                np_cam = np_cams[k]

                # get saliency
                bg_cue = cv2.imread(osp.join(args.dataset_root, "sal", args.sdnet_path, name + ".png"), cv2.IMREAD_GRAYSCALE)
                bg_cue = bg_cue.astype(np.float32)
                bg_cue = bg_cue / 255

                seg_mask = get_localization_cues_sec(np_cam, bg_cue, im_label, args.cam_thresh)

                # save mask
                write_mask = nd.zoom(seg_mask, (im_h / h, im_w / w), order=0)
                cv2.imwrite(osp.join(save_path, name + ".png"), write_mask)
