#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Author:  yaoqi (yaoqi_isee@zju.edu.cn)
# Created Date: 2019-12-02
# Modified By: yaoqi (yaoqi_isee@zju.edu.cn)
# Last Modified: 2019-12-06
# -----
# Copyright (c) 2019 Zhejiang University
"""
"""

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(self, conv5_dilation=1):
        super(VGG16, self).__init__()
        print("conv5_dilation = {}".format(conv5_dilation))
        
        # convolution layer
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=conv5_dilation, dilation=conv5_dilation)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=conv5_dilation, dilation=conv5_dilation)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=conv5_dilation, dilation=conv5_dilation)

        self.not_training = []
        self.from_scratch_layers = []

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))

        return x

    @property
    def out_channel(self):
        return 512


def vgg16(conv5_dilation=1):
    return VGG16(conv5_dilation)
