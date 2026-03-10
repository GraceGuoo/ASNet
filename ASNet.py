import os
import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import math,time

from resnet import Backbone_ResNet152

from torch.nn.modules.activation import PReLU
from toolbox.dct import MultiSpectralAttentionLayer as DCT
from toolbox.fam import FAM, FAM0


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.channela = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.channela(x)
        out = torch.mul(x, att)
        return out

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)
    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)
        return output

class TransBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=2, upsample=None):
        super(TransBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv3 = nn.ConvTranspose2d(planes, planes, kernel_size=2, stride=stride, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = PReLU(planes)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(inplanes, planes, kernel_size=2, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(planes),
        )
        self.stride = stride
        

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.upsample(x)
        out += residual
        out = self.relu(out)
        return out

class SA(nn.Module):
    def __init__(self, in_channel, norm_layer=nn.BatchNorm2d):
        super(SA, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = norm_layer(256)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)  # 256
        out2 = self.conv2(out1)
        w, b = out2[:, :256, :, :], out2[:, 256:, :, :]

        return F.relu(w * out1 + b, inplace=True)


class CrossAttention(nn.Module):
    def __init__(self, in_channel=256, ratio=8):
        super(CrossAttention, self).__init__()
        self.conv_query = nn.Conv2d(in_channel, in_channel // ratio, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channel, in_channel // ratio, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channel, in_channel, kernel_size=1)

    def forward(self, rgb, depth):
        bz, c, h, w = rgb.shape
        depth_q = self.conv_query(depth).view(bz, -1, h * w).permute(0, 2, 1)
        depth_k = self.conv_key(depth).view(bz, -1, h * w)
        mask = torch.bmm(depth_q, depth_k)  # bz, hw, hw
        mask = torch.softmax(mask, dim=-1)
        rgb_v = self.conv_value(rgb).view(bz, c, -1)
        feat = torch.bmm(rgb_v, mask.permute(0, 2, 1))  # bz, c, hw
        feat = feat.view(bz, c, h, w)
        return feat


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class CMAT(nn.Module):
    def __init__(self, in_channel, CA=False, ratio=8, c=256):
        super(CMAT, self).__init__()
        self.CA = CA
        if (in_channel < 256):
            c = in_channel

        if self.CA:
            print("Not USE CMAT")
        else:
            self.g1 = GhostModule(in_channel, c)
            self.g2 = GhostModule(in_channel, c)
            self.conv2 = nn.Conv2d(c, c, 3, 1, 1)
            self.conv3 = nn.Conv2d(c, c, 3, 1, 1)
            self.ca1 = ChannelAttention(c, ratio)
            self.ca2 = ChannelAttention(c, ratio)

    def forward(self, rgb, depth, beta, gamma, gate):

        if self.CA:
           pass
        else:
            rgb = self.g1(rgb)
            depth = self.g2(depth)
            w1 = self.conv2(rgb)
            w2 = self.conv3(depth)
            feat_1 = F.relu(self.ca1(w2 * rgb), inplace=True)
            feat_2 = F.relu(self.ca2(w1 * depth), inplace=True)

        out1 = rgb + gate * beta * feat_2
        out2 = depth + (1.0 - gate) * gamma * feat_1
       
        return out1, out2

class DCTCMAT(nn.Module):
    def __init__(self, in_channel, CA=True, ratio=8, c=256):
        super(DCTCMAT, self).__init__()
        self.CA = CA
        if (in_channel < 256):
            c = in_channel
        c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])
        if self.CA:
            self.dct1 = DCT(in_channel, c2wh[in_channel], c2wh[in_channel])
            self.dct2 = DCT(in_channel, c2wh[in_channel], c2wh[in_channel])
            self.conv1 = BasicConv2d(in_channel, c, 3, 1, 1)
            self.conv2 = BasicConv2d(in_channel, c, 3, 1, 1)
            self.sa1 = SA(c)
            self.sa2 = SA(c)
        else:
            print("Not use CrossAttention")

    def forward(self, rgb, depth, beta, gamma, gate):
        if self.CA:
            rgb = self.conv1(self.dct1(rgb))
            depth = self.conv2(self.dct2(depth))
            feat_1 = self.sa1(rgb)
            feat_2 = self.sa2(depth)
        else:
           pass
        out1 = rgb + gate * beta * feat_2
        out2 = depth + (1.0 - gate) * gamma * feat_1
        return out1, out2

class Fusion(nn.Module):
    def __init__(self, in_channel, out, norm_layer=nn.BatchNorm2d):
        super(Fusion, self).__init__()
        self.conv = nn.Conv2d(256, out, 3, 1, 1)
        self.conv0 = nn.Conv2d(out * 2, out, 3, 1, 1)
        self.bn0 = norm_layer(out)

    def forward(self, x1, x2, alpha, beta):
        alpha = self.conv(alpha)
        out1 = alpha * x1 + beta * (1.0 - alpha) * x2
        # out1 = x1 +  x2

        out2 = x1 * x2
        out = torch.cat((out1, out2), dim=1)
        out = F.relu(self.bn0(self.conv0(out)), inplace=True)

        return out


class ASNet(nn.Module):
    def __init__(self, n_classes):
        super(ASNet, self).__init__()
        (
            self.layer1_rgb,
            self.layer2_rgb,
            self.layer3_rgb,
            self.layer4_rgb,
            self.layer5_rgb,
        ) = Backbone_ResNet152(pretrained=True)

        self.rgbconv1 = BasicConv2d(64, 64, kernel_size=3, padding=1)  # 240 320
        self.rgbconv2 = BasicConv2d(256, 128, kernel_size=3, padding=1)  # 120 160
        self.rgbconv3 = BasicConv2d(512, 256, kernel_size=3, padding=1)  # 60 80
        self.rgbconv4 = BasicConv2d(1024, 256, kernel_size=3, padding=1)  # 30 40
        self.rgbconv5 = BasicConv2d(2048, 512, kernel_size=3, padding=1)  # 15 20
        channels = [128, 256, 256, 512]
       
        self.cmat5 = DCTCMAT(channels[3], True, 8)  # 8
        self.cmat4 = DCTCMAT(channels[2], True, 8)
        self.cmat3 = DCTCMAT(channels[1], True, 8)
        self.cmat2 = CMAT(channels[0], False, 8)
        self.cmat1 = CMAT(64, False, 8)

        # low- & high level
        self.fam54_1 = FAM(256, 256)
        self.fam43_1 = FAM(256, 256)
        self.fam32_1 = FAM0(128, 256, 128)
        self.fam21_1 = FAM0(64, 128)

        self.fam54_2 = FAM(256, 256)
        self.fam43_2 = FAM(256, 256)
        self.fam32_2 = FAM0(128, 256, 128)
        self.fam21_2 = FAM0(64, 128)

        self.fusion1 = Fusion(256, 256)
        self.fusion2 = Fusion(256, 128)
        self.fusion3 = Fusion(64, 64)
       
        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.gap2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[-1] * 2, 512),
            nn.Dropout(p=0.3),
            nn.ReLU(True),
            nn.Linear(512, 256 + 1),
            nn.Sigmoid(),
        )

        self.predsal = nn.Sequential(
            TransBottleneck(256, 128),
            TransBottleneck(128, 64),
            TransBottleneck(64, 32),
            TransBottleneck(32, 16),
            TransBottleneck(16, 8),
            nn.Conv2d(8, 2, kernel_size=3, padding=1, bias=True))
       
        self.semantic_pred2 = nn.Sequential(
            TransBottleneck(64, 32),
            nn.Conv2d(32, n_classes, kernel_size=3, padding=1))
      
        self.edge_pred = nn.Sequential(
            nn.Conv2d(n_classes, 2, kernel_size=3, padding=1))

        self.decoder1 = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 480 640
            TransBottleneck(128, 64),
            BasicConv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, n_classes, kernel_size=3, padding=1)
        )

    def forward(self, rgb, th):
        raw_size = rgb.size()[2:]
        bz = rgb.shape[0]

        x = rgb
        ir = th[:, :1, ...]
        ir = torch.cat((ir, ir, ir), dim=1)

        x1 = self.layer1_rgb(x)
        x2 = self.layer2_rgb(x1)
        x3 = self.layer3_rgb(x2)
        x4 = self.layer4_rgb(x3)
        x5 = self.layer5_rgb(x4)

        ir1 = self.layer1_rgb(ir)
        ir2 = self.layer2_rgb(ir1)
        ir3 = self.layer3_rgb(ir2)
        ir4 = self.layer4_rgb(ir3)
        ir5 = self.layer5_rgb(ir4)

        x1 = self.rgbconv1(x1)
        x2 = self.rgbconv2(x2)
        x3 = self.rgbconv3(x3)
        x4 = self.rgbconv4(x4)
        x5 = self.rgbconv5(x5)

        ir1 = self.rgbconv1(ir1)
        ir2 = self.rgbconv2(ir2)
        ir3 = self.rgbconv3(ir3)
        ir4 = self.rgbconv4(ir4)
        ir5 = self.rgbconv5(ir5)

        rgb_gap = self.gap1(x5)
        rgb_gap = rgb_gap.view(bz, -1)
        depth_gap = self.gap2(ir5)
        depth_gap = depth_gap.view(bz, -1)
        feat = torch.cat((rgb_gap, depth_gap), dim=1)
        feat = self.fc(feat)

        gate = feat[:, -1].view(bz, 1, 1, 1)

        alpha = feat[:, :256]
        alpha = alpha.view(bz, 256, 1, 1)

        out5_1, out5_2 = self.cmat5(x5, ir5, 1, 1, gate)
        de4_1, de4_2 = self.cmat4(x4, ir4, 1, 1, gate)
        de3_1, de3_2 = self.cmat3(x3, ir3, 1, 1, gate)
        de2_1, de2_2 = self.cmat2(x2, ir2, 1, 1, gate)
        de1_1, de1_2 = self.cmat1(x1, ir1, 1, 1, gate)

        out4_1 = self.fam54_1(de4_1, out5_1)
        out3_1 = self.fam43_1(de3_1, out4_1)
        out2_1 = self.fam32_1(de2_1, out3_1)
        out1_1 = self.fam21_1(de1_1, out2_1)

        out4_2 = self.fam54_2(de4_2, out5_2)
        out3_2 = self.fam43_2(de3_2, out4_2)
        out2_2 = self.fam32_2(de2_2, out3_2)
        out1_2 = self.fam21_2(de1_2, out2_2)

        sal = self.fusion1(out5_1, out5_2, alpha, gate)
        sal = self.predsal(sal)

        out2 = self.fusion2(out2_1, out2_2, alpha, gate)
        semantic = self.decoder1(out2)

        out3 = self.fusion3(out1_1, out1_2, alpha, gate)
        # print(x1.shape)

        semantic2 = self.semantic_pred2(out3)
        edge = self.edge_pred(semantic2)

        gate = gate.reshape(-1)
        
        return semantic, semantic2, sal, edge, gate


if __name__ == '__main__':
    # for MFNet dataset
    ASNet(9)
    # for PST900 dataset
    # ASNet(5)
