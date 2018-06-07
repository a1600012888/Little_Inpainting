import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
from torchvision import transforms, models
import torch.optim as optim
from collections import OrderedDict
from common import config
from tqdm import tqdm
import numpy as np
import math
import copy


class Self_Attn(nn.Module):
    '''
    input: batch_size x feature_depth x feature_size x feature_size
    attn_score: batch_size x feature_size x feature_size
    output: batch_size x feature_depth x feature_size x feature_size
    '''

    def __init__(self, b_size, imsize, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.b_size = b_size
        self.imsize = imsize
        self.in_dim = in_dim
        self.f_ = nn.Conv2d(in_dim, int(in_dim / 8)*(imsize ** 2), 1)
        self.g_ = nn.Conv2d(in_dim, int(in_dim / 8)*(imsize ** 2), 1)
        self.h_ = nn.Conv2d(in_dim, in_dim, 1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, pixel_wise=True):
        if pixel_wise:
            b_size = x.size(0)
            f_size = x.size(-1)

            f_x = self.f_(x)  # batch x in_dim/8*f*f x f_size x f_size
            g_x = self.g_(x)  # batch x in_dim/8*f*f x f_size x f_size
            h_x = self.h_(x).unsqueeze(2).repeat(1, 1, f_size ** 2, 1, 1).contiguous()\
                                                                .view(b_size, -1,f_size ** 2,f_size ** 2)  # batch x in_dim x f*f x f_size x f_size

            f_ready = f_x.contiguous().view(b_size, -1, f_size ** 2, f_size, f_size).permute(0, 1, 2, 4,
                                                                                             3)  # batch x in_dim*f*f x f_size2 x f_size1
            g_ready = g_x.contiguous().view(b_size, -1, f_size ** 2, f_size,
                                            f_size)  # batch x in_dim*f*f x f_size1 x f_size2

            attn_dist = torch.mul(f_ready, g_ready).sum(dim=1).contiguous().view(-1, f_size ** 2)  # batch*f*f x f_size1*f_size2
            attn_soft = F.softmax(attn_dist, dim=1).contiguous().view(b_size, f_size ** 2, f_size ** 2)  # batch x f*f x f*f
            attn_score = attn_soft.unsqueeze(1)  # batch x 1 x f*f x f*f

            self_attn_map = torch.mul(h_x, attn_score).sum(dim=3).contiguous().view(b_size, -1, f_size, f_size)  # batch x in_dim x f*f

            self_attn_map = self.gamma * self_attn_map + x

        return self_attn_map, attn_score





class _conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1):
        super(_conv_bn_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      dilation=dilation),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class _deconv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(_deconv_bn_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(up, self).__init__()
        self.deconv = _deconv_bn_relu(in_channels, mid_channels, kernel_size=4,
                                      stride=2, padding=1)
        self.conv = _conv_bn_relu(2 * mid_channels, out_channels, kernel_size=3,
                                  stride=1, padding=1)

    def forward(self, x1, x2):
        x1 = self.deconv(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(down, self).__init__()
        self.double_conv = nn.Sequential(
            _conv_bn_relu(in_channels, out_channels, kernel_size=3,
                          stride=2, padding=1),
            _conv_bn_relu(out_channels, out_channels, kernel_size=3,
                          stride=1, padding=1)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()

        #self.channels = [64, 128, 256, 128, 64, 32]
        self.channels = [12, 21, 42, 21, 12, 9]
        self.dilations = [2, 4, 8, 16]
        self.in_conv = _conv_bn_relu(3, self.channels[0], kernel_size=5,
                                     stride=1, padding=2)
        self.down1 = down(self.channels[0], self.channels[1])
        self.feature = nn.Sequential(OrderedDict([
            ('conv3_1', _conv_bn_relu(self.channels[1], self.channels[2], kernel_size=3,
                                      stride=2, padding=1)),
            ('conv3_2', _conv_bn_relu(self.channels[2], self.channels[2], kernel_size=3,
                                      stride=1, padding=1)),
            ('conv3_3', _conv_bn_relu(self.channels[2], self.channels[2], kernel_size=3,
                                      stride=1, padding=1))
        ]))
        for d in self.dilations:
            self.feature.add_module('conv3_dilated{}_1'.format(d),
                                    _conv_bn_relu(self.channels[2], self.channels[2],
                                                  kernel_size=3, stride=1, padding=d, dilation=d))
        self.feature.add_module('conv3_4', _conv_bn_relu(self.channels[2], self.channels[2], kernel_size=3,
                                                         stride=1, padding=1))
        self.feature.add_module('conv3_5', _conv_bn_relu(self.channels[2], self.channels[2], kernel_size=3,
                                                         stride=1, padding=1))
        '''
        for d in self.dilations:
            self.feature.add_module('conv3_dilated{}_2'.format(d),
                                    _conv_bn_relu(self.channels[2], self.channels[2],
                                                  kernel_size=3, stride=1, padding=d, dilation=d))
        self.feature.add_module('conv3_6', _conv_bn_relu(self.channels[2], self.channels[2], kernel_size=3,
                                                         stride=1, padding=1))
        self.feature.add_module('conv3_7', _conv_bn_relu(self.channels[2], self.channels[2], kernel_size=3,
                                                         stride=1, padding=1))
        '''
        self.up1 = up(self.channels[2], self.channels[3], self.channels[3])

        self.up2 = up(self.channels[3], self.channels[4], self.channels[5])
        self.out_conv = _conv_bn_relu(self.channels[5], 3, kernel_size=3,
                                      stride=1, padding=1)

        self.contrast_conv1 = _conv_bn_relu(self.channels[3], self.channels[3], kernel_size=3,
                                            stride=1, padding=1)
        self.contrast_conv2 = _conv_bn_relu(self.channels[3], self.channels[3], kernel_size=3,
                                            stride=1, padding=1)
        self.contrast_conv3 = _conv_bn_relu(self.channels[3], self.channels[3], kernel_size=3,
                                            stride=1, padding=1)
        #self.attn1 = Self_Attn(2, 64, self.channels[3], 'relu')
        #self.attn2 = Self_Attn(1, 128, self.channels[5], 'relu')
    def forward(self, x):
        self.x1 = self.in_conv(x)  # 64 * H * W
        # print('x1 size', x1.size())
        self.x2 = self.down1(self.x1)  # 128 * H/2 * W/2
        # print('x2 size', x2.size())
        self.x3 = self.feature(self.x2)  # 256 * H/4 * W/4
        # print('x3 size', x3.size())
        x = self.up1(self.x3, self.x2)

        x = self.contrast_conv1(x)
        x = self.contrast_conv2(x)
        x = self.contrast_conv3(x)

        #x, p1 = self.attn1(x)

        # print('x size', x.size())
        x = self.up2(x, self.x1)
        #print('testing attention shape: {}'.format(x.size()))
        # print('x size', x.size())
        #x, p2 = self.attn2(x)
        x = self.out_conv(x)
        # print('x size', x.size())
        return x


VGG = models.vgg19(pretrained=True).features.to(config.device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(config.device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(config.device)


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class vgg_for_style_transfer(nn.Module):
    def __init__(self):
        super(vgg_for_style_transfer, self).__init__()
        normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(config.device)
        self.f1 = nn.Sequential(normalization)
        self.f2 = nn.Sequential()
        self.f3 = nn.Sequential()
        self.f4 = nn.Sequential()
        self.f5 = nn.Sequential()

        i = 1  # increment every time we see a conv
        j = 1
        is_pool = 0
        for layer in VGG.children():
            if isinstance(layer, nn.Conv2d):
                name = 'conv_{}_{}'.format(i, j)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}_{}'.format(i, j)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                is_pool = 1
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}_{}'.format(i, j)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            if i == 1:
                self.f1.add_module(name, layer)
            elif i == 2:
                self.f2.add_module(name, layer)
            elif i == 3:
                self.f3.add_module(name, layer)
            elif i == 4:
                self.f4.add_module(name, layer)
            else:
                self.f5.add_module(name, layer)

            j += 1
            if is_pool:
                i += 1
                is_pool = 0
                j = 0

    def forward(self, x):
        feature1 = self.f1(x)
        feature2 = self.f2(feature1)
        feature3 = self.f3(feature2)
        feature4 = self.f4(feature3)
        feature5 = self.f5(feature4)

        return [feature1, feature2, feature3, feature4, feature5]


if __name__ == '__main__':
    net = vgg_for_style_transfer()
    print(net)

