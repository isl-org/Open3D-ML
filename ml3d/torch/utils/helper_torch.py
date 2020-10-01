import torch
import torch.nn as nn
import torch.nn.functional as F


class conv2d_transpose(nn.Module):

    def __init__(self,
                 batchNorm,
                 in_planes,
                 out_planes,
                 kernel_size=1,
                 stride=1,
                 activation=True):
        super(conv2d_transpose, self).__init__()
        self.conv = nn.ConvTranspose2d(in_planes,
                                       out_planes,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=(kernel_size - 1) // 2)
        self.biases = self.conv.bias
        self.weights = self.conv.weight
        self.batchNorm = batchNorm

        self.batch_normalization = nn.BatchNorm2d(out_planes,
                                                  momentum=0.01,
                                                  eps=1e-6)

        if activation:
            self.activation_fn = nn.LeakyReLU(0.2)
        else:
            self.activation_fn = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        if self.batchNorm:
            x = self.batch_normalization(x)
        x = self.activation_fn(x)
        return x


class conv2d(nn.Module):

    def __init__(self,
                 batchNorm,
                 in_planes,
                 out_planes,
                 kernel_size=1,
                 stride=1,
                 activation=True):
        super(conv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes,
                              out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(kernel_size - 1) // 2)
        self.biases = self.conv.bias
        self.weights = self.conv.weight
        self.batchNorm = batchNorm
        if self.batchNorm:
            self.batch_normalization = nn.BatchNorm2d(out_planes,
                                                      momentum=0.01,
                                                      eps=1e-6)

        if activation:
            self.activation_fn = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation_fn = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        if self.batchNorm:
            x = self.batch_normalization(x)
        x = self.activation_fn(x)
        return x
