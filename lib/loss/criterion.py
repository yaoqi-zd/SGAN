import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from .loss import Seedloss, Boundaryloss
import scipy.ndimage as nd
from ipdb import set_trace


class CritetionBalancedSoftMargin(nn.Module):
    def __init__(self):
        super(CritetionBalancedSoftMargin, self).__init__()
        self.balanced_loss = BalancedSoftMarginLoss()

    def forward(self, score, target):
        return self.balanced_loss(score, target)

class CriterionDSRG(nn.Module):
    """
        Compute dsrg loss (balanced seed loss + boundary loss)
    """
    def __init__(self, ignore_index=201):
        super(CriterionDSRG, self).__init__()
        self.ignore_index = ignore_index
        self.seed_criterion = Seedloss(ignore_label=self.ignore_index)
        self.boundary_criterion = Boundaryloss()

    def forward(self, inputs, target):
        images = inputs[0]
        preds = inputs[1]
        # compute seed loss
        h, w = target.size(2), target.size(3)

        seed_loss = self.seed_criterion(preds, target)

        # compute boundary loss
        bound_loss = self.boundary_criterion(images, preds)

        return [seed_loss, bound_loss]

class CriterionBSL(nn.Module):
    """compute balanced seed loss"""
    def __init__(self, ignore_index=201):
        super(CriterionBSL, self).__init__()
        self.ignore_index = ignore_index
        self.seed_criterion = Seedloss(ignore_label=self.ignore_index)

    def forward(self, pred, target):
        return self.seed_criterion(pred, target)


class CriterionBound(nn.Module):
    """compute boundary loss"""
    def __init__(self):
        super(CriterionBound, self).__init__()
        self.boundary_criterion = Boundaryloss()

    def forward(self, pred, image):
        return self.boundary_criterion(image, pred)







