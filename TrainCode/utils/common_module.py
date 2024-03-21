# encoding: utf-8
"""
 *@Author: Benjay·Shaw
 *@CreateTime: 2022/6/14 上午8:59
 *@LastEditors: Benjay·Shaw
 *@LastEditTime:2022/6/14 上午8:59
 *@Description: 遥感解译common module类
 """
from utils.attention import *
import torch
from torch import nn
import torch.nn.functional as F


class SPPBlock(nn.Module):
    def __init__(self, in_channels):
        super(SPPBlock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=(5, 5), stride=5)
        # self.pool4 = nn.MaxPool2d(kernel_size=(6, 6) stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        in_channels, h, w = x.size(1), x.size(2), x.size(3)

        layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        # self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([layer1, layer2, layer3,
                         # self.layer4,
                         x], 1)

        return out


class DBlockMoreDilate(nn.Module):
    def __init__(self, channel):
        super(DBlockMoreDilate, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = self.relu(self.dilate1(x))
        dilate2_out = self.relu(self.dilate2(dilate1_out))
        dilate3_out = self.relu(self.dilate3(dilate2_out))
        out = x + dilate1_out + dilate2_out + dilate3_out
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.de_conv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.de_conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

