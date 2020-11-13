import cv2
import functools
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, gradcheck

from coordconv import CoordConv
from deformconv import *

LEAKY_FACTOR = 0.2
MULT_FACTOR = 1

class GridSamplerFunction(Function):
    @staticmethod
    def forward(ctx, img, kernels, offset_h, offsets_v, offset_unit, padding, downscale_factor):
        assert isinstance(downscale_factor, int)
        assert isinstance(padding, int)

        ctx.padding = padding
        ctx.offset_unit = offset_unit

        b, c, h, w =  img.size()
        assert h // downscale_factor == kernels.size(2)
        assert w // downscale_factor == kernels.size(3)

        img = nn.ReflectionPad2d(padding)(img)

        output = img.new(b, c, h // downscale_factor, w // downscale_factor).zero_()
        forward(img, kernels, offset_h, offsets_v, offset_unit, padding, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

class Downsampler(nn.Module):
    def __init__(self, ds, k_size):
        super(Downsampler, self).__init__()
        self.ds = ds
        self.k_size = k_size

    def forward(self, img, kernels, offsets_h, offsets_v, offset_unit):
        assert self.k_size ** 2 == kernels.size(1)
        return GridSamplerFunction.apply(img, kernels, offsets_h, offsets_v, offset_unit, self.k_size // 2, self.ds)


class PixelUnShuffle(nn.Module):
    """
    Inverse process of pytorch pixel shuffle module
    """
    def __init__(self, down_scale):
        """
        :param down_scale: int, down scale factor
        """
        super(PixelUnShuffle, self).__init__()

        if not isinstance(down_scale, int):
            raise ValueError('Down scale factor must be a integer number')
        self.down_scale = down_scale

    def forward(self, input):
        """
        :param input: tensor of shape (batch size, channels, height, width)
        :return: tensor of shape(batch size, channels * down_scale * down_scale, height / down_scale, width / down_scale)
        """
        b, c, h, w = input.size()
        assert h % self.down_scale == 0
        assert w % self.down_scale == 0

        oc = c * self.down_scale ** 2
        oh = int(h / self.down_scale)
        ow = int(w / self.down_scale)

        output_reshaped = input.reshape(b, c, oh, self.down_scale, ow, self.down_scale)
        output = output_reshaped.permute(0, 1, 3, 5, 2, 4).reshape(b, oc, oh, ow)

        return output


class DownsampleBlock(nn.Module):
    def __init__(self, scale, input_channels, output_channels, ksize=1):
        super(DownsampleBlock, self).__init__()
        self.downsample = nn.Sequential(
            PixelUnShuffle(scale),
            nn.Conv2d(input_channels * (scale ** 2), output_channels, kernel_size=ksize, stride=1, padding=ksize//2),
            nn.BatchNorm2d(output_channels) # add batchnorm
        )

    def forward(self, input):
        return self.downsample(input)


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, channels, ksize=3,
                 use_instance_norm=False, affine=False):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.ksize = ksize
        padding = self.ksize // 2
        if use_instance_norm:
            self.transform = nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.Conv2d(input_channels, channels, kernel_size=self.ksize, stride=1),
                nn.InstanceNorm2d(channels, affine=affine),
                nn.LeakyReLU(0.2),
                nn.ReflectionPad2d(padding),
                nn.Conv2d(channels, channels, kernel_size=self.ksize, stride=1),
                nn.InstanceNorm2d(channels)
            )
        else:
            self.transform = nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.Conv2d(input_channels, channels, kernel_size=self.ksize, stride=1),
                nn.LeakyReLU(0.2),
                nn.ReflectionPad2d(padding),
                nn.Conv2d(channels, channels, kernel_size=self.ksize, stride=1),
            )

    def forward(self, input):
        return input + self.transform(input) * MULT_FACTOR

class DSN(nn.Module):
    def __init__(self, k_size, input_channels=6, scale=4):
        super(DSN, self).__init__()

        self.k_size = k_size

        # self.sub_mean = MeanShift(1)
        self.coordconv = CoordConv(input_channels, 64, with_r=True)
        self.dc = DeformConv2d(64, 64, 3, padding=1, modulation=True)

        self.ds_1 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64), # add batchnorm
            nn.LeakyReLU(LEAKY_FACTOR)
        )

        self.ds_2 = DownsampleBlock(2, 64, 128, ksize=1)
        self.ds_4 = DownsampleBlock(2, 128, 128, ksize=1)


        res_4 = list()
        for idx in range(5):
            res_4 += [ResidualBlock(128, 128)]
        self.res_4 = nn.Sequential(*res_4)

        self.ds_8 = DownsampleBlock(2, 128, 256)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.coordconv(x)
        x = self.dc(x)

        x = self.ds_1(x)
        x = self.ds_2(x) # x 1/2
        x = self.ds_4(x) # x 1/4
        x = x + self.res_4(x)
        x = self.ds_8(x) # x 1/8

        return x

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class EDSR(nn.Module):
    def __init__(self, n_resblocks=16, n_feats=64, scale=8, conv=default_conv):
        super(EDSR, self).__init__()


        kernel_size = 3
        act = nn.ReLU(True)

        self.coordconv1 = CoordConv(256, 3, with_r=True)
        self.coordconv2 = CoordConv(3, 3, with_r=True)

        # define head module
        m_head = [conv(3, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=0.1
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.coordconv1(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)
        x = self.coordconv2(x)

        return x


class BILM(nn.Module):
    def __init__(self):
        super(BILM, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, feat):
        pos_sig = torch.sigmoid(feat)
        neg_sig = -1 * pos_sig

        pos_sig = self.maxpool1(pos_sig)
        neg_sig = self.maxpool2(neg_sig)
        sum_sig = pos_sig + neg_sig

        x = feat * sum_sig

        return x

class CAR(nn.Module):
    def __init__(self, classes=1, is_training=True, criterion=nn.BCEWithLogitsLoss()):#criterion=nn.BCELoss()):
        super(CAR, self).__init__()

        self.criterion = criterion
        self.num_class = classes
        self.is_training = is_training

        self.dsn = DSN(k_size=3)

        self.bilm1 = BILM()
        self.bilm2 = BILM()

        self.downsampler = Downsampler(4, 3)
        self.edsr = EDSR()
        self.output = nn.Conv2d(256, 3, 3, 1, 1)
        self.output = nn.Sequential(nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(3),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout2d(p=0.2),
                                    nn.Conv2d(3, classes, kernel_size=1, stride=1, padding=0))

    def forward(self, x, y=None):
        # encoder
        x = self.dsn(x)

        x = self.bilm1(x)

        # decoder
        x = self.edsr(x)

        x = self.output(x)

        x = torch.sigmoid(x)

        if self.training:
            loss = self.criterion(x, y)

            return x, loss
        else:
            return x