import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.ndimage as nd
from ipdb import set_trace
import krahenbuhl2013
from multiprocessing import Pool
import matplotlib.pyplot as plt
import cv2, os

import pickle

def crf_single(inp):
    im, prob = inp[0], inp[1]
    im = np.squeeze(im, axis=0)
    prob = np.squeeze(prob, axis=0)
    return krahenbuhl2013.CRF(im, prob, scale_factor=12.0)

class Boundaryloss(nn.Module):
    def __init__(self):
        super(Boundaryloss, self).__init__()
        self.eps = 1e-5
        self.scale = 1

    def crf(self, images, probs):
        np_ims = np.transpose(images.data.cpu().numpy(), [0, 2, 3, 1])  # (n, H, W, c)
        np_probs = np.transpose(probs.data.cpu().numpy(), [0, 2, 3, 1]) # (n, h, w, c)
        
        im_h, im_w = np_ims.shape[1:3]
        h, w = np_probs.shape[1:3]
        np_ims = nd.zoom(np_ims, (1.0, h / im_h, w / im_w, 1.0), order=1)

        batch_size = np_ims.shape[0]
        list_crf = []
        for k in range(batch_size):
            list_crf.append(krahenbuhl2013.CRF(np_ims[k], np_probs[k], scale_factor=12.0))
        crf_prob = np.stack(list_crf, axis=0)

        crf_prob = np.transpose(crf_prob, [0, 3, 1, 2])
        crf_prob[crf_prob < self.eps] = self.eps
        crf_prob = crf_prob / np.sum(crf_prob, axis=1, keepdims=True)

        result = torch.from_numpy(crf_prob).float().cuda(probs.get_device())

        return result

    def my_softmax(self, score, dim=1):
        probs = torch.clamp(F.softmax(score, dim), self.eps, 1)
        probs = probs / torch.sum(probs, dim=dim, keepdim=True)
        return probs

    def forward(self, images, predict, out_prob=False):
        """
        compute the constrain-to-boundary loss
        :param images: (n, 3, H, W)
        :param predict: (n, c, h, w)
        :return:
        """

        probs = self.my_softmax(predict, dim=1)
        probs_smooth = self.crf(images, probs)
        loss = torch.mean(torch.sum(probs_smooth * torch.log(torch.clamp(probs_smooth / probs, 0.05, 20)), dim=1))

        if out_prob:
            return loss, probs_smooth

        return loss

class Seedloss(nn.Module):
    def __init__(self, ignore_label=21):
        super(Seedloss, self).__init__()
        self.ignore_label = ignore_label
        self.eps = 1e-5

    def my_softmax(self, score, dim=1):
        probs = torch.clamp(F.softmax(score, dim), self.eps, 1)
        probs = probs / torch.sum(probs, dim=dim, keepdim=True)
        return probs

    def forward(self, predict, target):
        """
        compute balanced seed loss
        :param predict: (n, c, h, w)
        :param target: (n, c, h, w)
        :return:
        """
        assert not target.requires_grad
        target = target.to(torch.float)
        
        # set_trace()
        assert torch.sum(torch.isinf(predict)) == 0
        assert torch.sum(torch.isnan(predict)) == 0 

        input_log_prob = torch.log(self.my_softmax(predict, dim=1))

        assert torch.sum(torch.isnan(input_log_prob)) == 0 

        fg_prob = input_log_prob[:, 1:, :, :]
        fg_label = target[:, 1:, :, :]
        fg_count = torch.sum(fg_label, dim=(1, 2, 3)) + self.eps

        bg_prob = input_log_prob[:, 0:1, :, :]
        bg_label = target[:, 0:1, :, :]
        bg_count = torch.sum(bg_label, dim=(1, 2, 3)) + self.eps

        loss_fg = torch.sum(fg_label * fg_prob, dim=(1, 2, 3))
        loss_fg = -1 * torch.mean(loss_fg / fg_count)

        loss_bg = torch.sum(bg_label * bg_prob, dim=(1, 2, 3))
        loss_bg = -1 * torch.mean(loss_bg / bg_count)

        total_loss = loss_bg + loss_fg

        assert torch.sum(torch.isnan(total_loss)) == 0, \
            "fg_loss: {} fg_count: {} bg_loss: {} bg_count: {}".format(loss_fg, fg_count, loss_bg, bg_count)

        return total_loss


