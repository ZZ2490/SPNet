# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import cv2

import os, argparse

from Encoder import Encoder


import pdb

class Singe_prototype(nn.Module):


    def __init__(self, in_c, num_p):
        super(Singe_prototype, self).__init__()
        self.num_cluster = num_p
        self.netup = torch.nn.Sequential(
                torch.nn.Conv2d(in_c, num_p, 3, padding=1)
                )
        self.centroids = torch.nn.Parameter(torch.rand(num_p, in_c))   #生成（24, 256)可训练学习的张量

        self.upfc = torch.nn.Linear(num_p*in_c, in_c)

        self.transform = torch.nn.Sequential(
            nn.Conv2d(2*in_c, in_c, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_c, in_c, kernel_size=1),
            nn.ReLU(inplace=False),
            )

    def UP(self, scene):
        x = scene

        N, C, W, H = x.shape[0:]

        x = F.normalize(x, p=2, dim=1)          #对x做正则化，除2范数
        soft_assign = self.netup(x)                #通道数变为24

        soft_assign = F.softmax(soft_assign, dim=1)  #通道注意力机制
        soft_assign = soft_assign.view(soft_assign.shape[0], soft_assign.shape[1], -1)
        #调整图的大小
        x_flatten = x.view(N, C, -1)

        centroid = self.centroids       #生成（24, 256)可训练学习的张量

        x1 = x_flatten.expand(self.num_cluster, -1, -1, -1).permute(1, 0, 2, 3) #对维度进行扩展
        x2 = centroid.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)#在0处增加一个维度

        residual = x1 - x2
        residual = residual * soft_assign.unsqueeze(2)
        up = residual.sum(dim=-1)

        up = F.normalize(up, p=2, dim=2)
        up = up.view(x.size(0), -1)
        up = F.normalize(up, p=2, dim=1)

        up = self.upfc(up).unsqueeze(2).unsqueeze(3).repeat(1,1,W,H)

        return up, centroid

    def forward(self, feature):

        up, centroid = self.UP(feature)

        map_glo = torch.unsqueeze(torch.mean(feature, 1), 1)
        map_glo = (map_glo - map_glo.min()) / (map_glo.max() - map_glo.min() + 1e-8)
        map_glo = F.interpolate(map_glo, size=(96, 96), mode='bilinear', align_corners=True)
        map_glo = np.squeeze(torch.Tensor(map_glo.cpu().data.numpy()).cpu().data.numpy())
        heatmap = cv2.applyColorMap(np.uint8(255 * map_glo), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join('./output/r_f/', 'f1' + '.png'), heatmap)

        new_feature = torch.cat((feature, up), dim=1)
        new_feature = self.transform(new_feature)

        map_glo = torch.unsqueeze(torch.mean(new_feature, 1), 1)
        map_glo = (map_glo - map_glo.min()) / (map_glo.max() - map_glo.min() + 1e-8)
        map_glo = F.interpolate(map_glo, size=(96, 96), mode='bilinear', align_corners=True)
        map_glo = np.squeeze(torch.Tensor(map_glo.cpu().data.numpy()).cpu().data.numpy())
        heatmap = cv2.applyColorMap(np.uint8(255 * map_glo), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join('./output/r_f/', 'f2' + '.png'), heatmap)

        return new_feature

