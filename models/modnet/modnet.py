from typing import Tuple, Any, List

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .mobilenetv2 import MobileNetV2Backbone


class IBNorm(nn.Module):
    """ Combine Instance Norm and Batch Norm into One Layer"""

    def __init__(self, in_channels: int):
        super(IBNorm, self).__init__()

        self.bnorm_channels = int(in_channels / 2)
        self.inorm_channels = in_channels - self.bnorm_channels

        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)

    def forward(self, input: Tensor) -> Tensor:
        bn_x = self.bnorm(input[:, :self.bnorm_channels, ...].contiguous())
        in_x = self.inorm(input[:, self.bnorm_channels:, ...].contiguous())

        return torch.cat((bn_x, in_x), 1)


class Conv2dIBNormRelu(nn.Module):
    """ Convolution + IBNorm + ReLu"""

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...] = (1, 1),
                 padding: Tuple[int, ...] = (0, 0),
                 dilation: Tuple[int, ...] = (1, 1),
                 groups: int = 1, bias: bool = True,
                 with_ibn: bool = True, with_relu: bool = True):

        super(Conv2dIBNormRelu, self).__init__()

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups, bias=bias
            )
        ]

        if with_ibn:
            layers.append(IBNorm(out_channels))
        if with_relu:
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class SEBlock(nn.Module):
    """ SE Block Proposed in https://arxiv.org/pdf/1709.01507.pdf"""

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 1):
        super(SEBlock, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, int(in_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels // reduction), out_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)

        return x * w.expand_as(x)


class LRBranch(nn.Module):
    """ Low Resolution Branch of MODNet"""

    def __init__(self, backbone: nn.Module):
        super(LRBranch, self).__init__()

        enc_channels = backbone.enc_channels

        self.backbone = backbone
        self.se_block = SEBlock(enc_channels[4], enc_channels[4], reduction=4)

        self.conv_lr16x = Conv2dIBNormRelu(
            in_channels=enc_channels[4],
            out_channels=enc_channels[3],
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2)
        )

        self.conv_lr8x = Conv2dIBNormRelu(
            in_channels=enc_channels[3],
            out_channels=enc_channels[2],
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2)
        )

        self.conv_lr = Conv2dIBNormRelu(
            in_channels=enc_channels[2],
            out_channels=1,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            with_ibn=False, with_relu=False
        )

    def forward(self, img: Tensor) -> Tuple[Any, List[Tensor]]:
        enc_features = self.backbone.forward(img)
        enc2x, enc4x, enc32x = enc_features[0], enc_features[1], enc_features[4]

        enc32x = self.se_block(enc32x)

        lr16x = F.interpolate(enc32x, scale_factor=2, mode='bilinear', align_corners=False)
        lr16x = self.conv_lr16x(lr16x)

        lr8x = F.interpolate(lr16x, scale_factor=2, mode='bilinear', align_corners=False)
        lr8x = self.conv_lr8x(lr8x)

        return lr8x, [enc2x, enc4x]


class HRBranch(nn.Module):
    """ High Resolution Branch of MODNet"""

    def __init__(self, hr_channels: int, enc_channels: List[int]):
        super(HRBranch, self).__init__()

        self.tohr_enc2x = Conv2dIBNormRelu(enc_channels[0], hr_channels, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_enc2x = Conv2dIBNormRelu(hr_channels + 3, hr_channels, (3, 3), stride=(2, 2), padding=(1, 1))

        self.tohr_enc4x = Conv2dIBNormRelu(enc_channels[1], hr_channels, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_enc4x = Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, (3, 3), stride=(1, 1), padding=(1, 1))

        self.conv_hr4x = nn.Sequential(
            Conv2dIBNormRelu(3 * hr_channels + 3, 2 * hr_channels, (3, 3), stride=(1, 1), padding=(1, 1)),
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, (3, 3), stride=(1, 1), padding=(1, 1)),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, (3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.conv_hr2x = nn.Sequential(
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, (3, 3), stride=(1, 1), padding=(1, 1)),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, (3, 3), stride=(1, 1), padding=(1, 1)),
            Conv2dIBNormRelu(hr_channels, hr_channels, (3, 3), stride=(1, 1), padding=(1, 1)),
            Conv2dIBNormRelu(hr_channels, hr_channels, (3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.conv_hr = nn.Sequential(
            Conv2dIBNormRelu(hr_channels + 3, hr_channels, (3, 3), stride=(1, 1), padding=(1, 1)),
            Conv2dIBNormRelu(
                in_channels=hr_channels, out_channels=1,
                kernel_size=(1, 1), stride=(1, 1),
                padding=(0, 0), with_ibn=False, with_relu=False
            ),
        )

    def forward(self, img: Tensor,
                enc2x: Tensor,
                enc4x: Tensor,
                lr8x: Tensor) -> Tensor:

        img2x = F.interpolate(img, scale_factor=1 / 2, mode='bilinear',
                              align_corners=False, recompute_scale_factor=True)

        img4x = F.interpolate(img, scale_factor=1 / 4, mode='bilinear',
                              align_corners=False, recompute_scale_factor=True)

        enc2x = self.tohr_enc2x(enc2x)
        hr4x = self.conv_enc2x(torch.cat((img2x, enc2x), dim=1))

        enc4x = self.tohr_enc4x(enc4x)
        hr4x = self.conv_enc4x(torch.cat((hr4x, enc4x), dim=1))

        lr4x = F.interpolate(lr8x, scale_factor=2, mode='bilinear', align_corners=False)
        hr4x = self.conv_hr4x(torch.cat((hr4x, lr4x, img4x), dim=1))

        hr2x = F.interpolate(hr4x, scale_factor=2, mode='bilinear', align_corners=False)
        hr2x = self.conv_hr2x(torch.cat((hr2x, enc2x), dim=1))

        return hr2x


class FusionBranch(nn.Module):
    """ Fusion Branch of MODNet"""

    def __init__(self, hr_channels: int, enc_channels: List[int]):
        super(FusionBranch, self).__init__()

        self.conv_lr4x = Conv2dIBNormRelu(enc_channels[2], hr_channels, (5, 5), stride=(1, 1), padding=(2, 2))

        self.conv_f2x = Conv2dIBNormRelu(2 * hr_channels, hr_channels, (3, 3), stride=(1, 1), padding=(1, 1))

        self.conv_f = nn.Sequential(
            Conv2dIBNormRelu(
                in_channels=hr_channels + 3, out_channels=int(hr_channels / 2),
                kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ),
            Conv2dIBNormRelu(
                in_channels=int(hr_channels / 2), out_channels=1, kernel_size=(1, 1),
                stride=(1, 1), padding=(0, 0), with_ibn=False, with_relu=False
            ),
        )

    def forward(self, img: Tensor, lr8x: Tensor, hr2x: Tensor) -> Tensor:
        lr4x = F.interpolate(lr8x, scale_factor=2, mode='bilinear', align_corners=False)
        lr4x = self.conv_lr4x(lr4x)

        lr2x = F.interpolate(lr4x, scale_factor=2, mode='bilinear', align_corners=False)

        f2x = self.conv_f2x(torch.cat((lr2x, hr2x), dim=1))

        f = F.interpolate(f2x, scale_factor=2, mode='bilinear', align_corners=False)
        f = self.conv_f(torch.cat((f, img), dim=1))

        matte = torch.sigmoid(f)
        return matte


class MODNet(nn.Module):
    """ Architecture of MODNet"""

    def __init__(self, in_channels: int = 3, hr_channels: int = 32):
        super(MODNet, self).__init__()

        self.in_channels = in_channels
        self.hr_channels = hr_channels

        self.backbone = MobileNetV2Backbone(self.in_channels)

        self.lr_branch = LRBranch(self.backbone)
        self.hr_branch = HRBranch(self.hr_channels, self.backbone.enc_channels)
        self.f_branch = FusionBranch(self.hr_channels, self.backbone.enc_channels)

    def forward(self, img: Tensor) -> Tuple[Tensor, Tensor]:
        lr8x, [enc2x, enc4x] = self.lr_branch(img)
        hr2x = self.hr_branch(img, enc2x, enc4x, lr8x)

        matte = self.f_branch(img, lr8x, hr2x)
        return matte
