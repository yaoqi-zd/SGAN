import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import pickle
from ipdb import set_trace
import scipy.ndimage as nd
import matplotlib.pyplot as plt
# import backbone
from lib.network import backbone


def norm(x):
    min_x = np.min(x)
    x = x - min_x
    return x / np.sum(x)

def masked_softmax(x, mask, dim=-1, epsilon=1e-5):
    # x: [N, HW, HW]
    max_x, _ = torch.max(x, dim=-1, keepdim=True)
    exps = torch.exp(x - max_x)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return masked_exps / masked_sums

def masked_normalize(x, mask, eps=1e-5):
    min_v, _ = torch.min(x, dim=-1, keepdim=True)
    x1 = (x - min_v) * mask
    sum_x1 = torch.sum(x1, dim=-1, keepdim=True)
    return x1 / (sum_x1 + eps)

def softmax(x, dim=-1, epsilon=1e-5):
    max_x, _ = torch.max(x, dim=dim, keepdim=True)
    exps = torch.exp(x - max_x)
    sums = exps.sum(dim, keepdim=True) + epsilon
    return exps / sums

class SGAN(nn.Module):
    def __init__(self, backbone_name=None):
        super(SGAN, self).__init__()
        assert hasattr(backbone, backbone_name)
        self.backbone = getattr(backbone, backbone_name)()

        self.from_scratch_layers = []

        # attention layer
        self.in_channel, self.out_channel = self.backbone.out_channel, self.backbone.out_channel
        self.key_channel, self.query_channel, self.value_channel = self.in_channel // 2, self.in_channel // 2, self.in_channel
        self.f_key = nn.Conv2d(self.in_channel, self.key_channel, 1)
        self.f_query = nn.Conv2d(self.in_channel, self.query_channel, 1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.from_scratch_layers.extend([self.f_key, self.f_query])

        # image classification branch
        self.fc6_CAM = nn.Conv2d(self.out_channel, 1024, 3, padding=1)
        self.drop6 = nn.Dropout2d(p=0.5)
        self.fc8 = nn.Conv2d(1024, 20, 1, bias=False)
        self.from_scratch_layers.extend([self.fc6_CAM, self.fc8])

        # seed segmentation branch
        self.fc9 = nn.Conv2d(self.out_channel, 20, 1)
        self.from_scratch_layers.append(self.fc9)

    def forward_fc(self, x):
        x = F.relu(self.fc6(x))
        x = self.drop6(x)
        x = F.relu(self.fc7(x))
        x = self.drop7(x)

        return x

    def attention_module(self, x, fg_sim, bg_sim):
        """Build an attention module on top of the extract feature map"""
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        # value = self.f_value(x).view(batch_size, self.value_channel, -1)
        value = x.view(batch_size, self.value_channel, -1)
        value = value.permute(0, 2, 1)

        query = self.f_query(x).view(batch_size, self.query_channel, -1)
        query = query.permute(0, 2, 1)

        key = self.f_key(x).view(batch_size, self.key_channel, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channel ** -.5) * sim_map

        sim_map = sim_map * fg_sim
        sim_map = masked_normalize(sim_map, fg_sim)
        # sim_map = masked_softmax(sim_map, fg_sim)
        sim_map = sim_map + bg_sim

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channel, *x.size()[2:])

        fuse = self.gamma * context + x

        return fuse

    def forward_sim_map(self, x, fg_sim, bg_sim):
        x = self.backbone(x)

        batch_size, h, w = x.size(0), x.size(2), x.size(3)
    
        new_size = round(321 / 4 + 0.5)

        query = self.f_query(x)
        key = self.f_key(x)
        query = F.upsample(query, size=(new_size, new_size), mode="bilinear", align_corners=True)
        key = F.upsample(key, size=(new_size, new_size), mode="bilinear", align_corners=True)

        query = query.view(batch_size, self.query_channel, -1).permute(0, 2, 1)
        key = key.view(batch_size, self.key_channel, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channel ** -.5) * sim_map

        sim_map = sim_map * fg_sim
        sim_map = masked_normalize(sim_map, fg_sim)
        sim_map = sim_map + bg_sim

        return sim_map

    def forward_cam(self, x, fg_sim, bg_sim):
        conv = self.backbone(x)
        fuse = self.attention_module(conv, fg_sim, bg_sim)

        feat = self.drop6(F.relu(self.fc6_CAM(fuse)))
        cam = self.fc8(feat)

        seg = self.fc9(fuse)

        return conv, seg, cam

    def forward(self, x, fg_sim, bg_sim):
        conv = self.backbone(x)
        fuse = self.attention_module(conv, fg_sim, bg_sim)

        # classification branch
        feat = self.drop6(F.relu(self.fc6_CAM(fuse)))
        cam = self.fc8(feat)
        score = F.avg_pool2d(cam, kernel_size=(cam.size(2), cam.size(3)), padding=0)
        # score = self.global_dynamic_pooling(cam, feat)

        score = score.view(score.size(0), -1)

        seg = self.fc9(fuse)

        return [score, seg]

    def initialize(self, pretrained):
        """initialize the backbone parameters"""
        model_dict = self.backbone.state_dict()
        weight_dict = torch.load(pretrained)
        pretrained_dict = {k:v for k, v in weight_dict.items() if k in model_dict}
        for k in sorted(pretrained_dict.keys()):
            print("loading pretrained weights: {:<20s} shape: {}".format(k, pretrained_dict[k].shape))
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)

    def get_param_groups(self):
        groups = ([], [], [], []) # 1, 2, 10, 20

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight is not None and m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

            if isinstance(m, nn.Linear):
                if m.weight is not None and m.weight.requires_grad:
                    groups[2].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    groups[3].append(m.bias)

        groups[2].append(self.gamma)

        return groups


if __name__ == '__main__':
    net = VGG16()
    x = torch.rand(size=(2, 3, 321, 321))
    fg_sim = torch.ones(size=(2, 1681, 1681))
    bg_sim = torch.ones(size=(2, 1681, 1681))
    _, _, out = net.forward_cam(x, fg_sim, bg_sim)
    print(out.size())
