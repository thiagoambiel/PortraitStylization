""" This file is adapted from https://github.com/ZHKKKe/MODNet"""
from typing import Tuple, List

from torch import nn, Tensor
from functools import reduce


class InvertedResidual(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 expansion: int,
                 stride: Tuple[int, ...],
                 dilation: Tuple[int, ...] = (1, 1)):

        super(InvertedResidual, self).__init__()

        self.stride = stride
        self.use_res_connect = self.stride == (1, 1) and in_channels == out_channels

        hidden_dim = round(in_channels * expansion)
        if expansion == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=(3, 3),
                    stride=self.stride,
                    padding=(1, 1),
                    groups=hidden_dim,
                    dilation=dilation,
                    bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=out_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=(3, 3),
                    stride=self.stride,
                    padding=(1, 1),
                    groups=hidden_dim,
                    dilation=dilation,
                    bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=out_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, input: Tensor) -> Tensor:
        if self.use_res_connect:
            return input + self.conv(input)
        else:
            return self.conv(input)


class MobileNetV2(nn.Module):
    def __init__(self, in_channels: int, alpha: float = 1.0, expansion: int = 6):
        super(MobileNetV2, self).__init__()
        self.in_channels = in_channels

        input_channel = 32
        last_channel = 1280

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [expansion, 24, 2, 2],
            [expansion, 32, 3, 2],
            [expansion, 64, 4, 2],
            [expansion, 96, 3, 1],
            [expansion, 160, 3, 2],
            [expansion, 320, 1, 1],
        ]

        # building first layer
        input_channel = self._make_divisible(input_channel * alpha, 8)
        self.last_channel = self._make_divisible(last_channel * alpha, 8) if alpha > 1.0 else last_channel

        self.features = [self.conv_bn_relu(
            in_channels=self.in_channels,
            out_channels=input_channel,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1)
        )]

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = self._make_divisible(int(c * alpha), 8)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(
                        in_channels=input_channel,
                        out_channels=output_channel,
                        stride=s,
                        expansion=t
                    ))
                else:
                    self.features.append(InvertedResidual(
                        in_channels=input_channel,
                        out_channels=output_channel,
                        stride=(1, 1),
                        expansion=t
                    ))

                input_channel = output_channel

        # building last several layers
        self.features.append(self.conv_bn_relu(
            in_channels=input_channel,
            out_channels=self.last_channel,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0)
        ))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

    @staticmethod
    def conv_bn_relu(in_channels: int,
                     out_channels: int,
                     kernel_size: Tuple[int, ...],
                     stride: Tuple[int, ...],
                     padding: Tuple[int, ...]) -> nn.Module:

        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    @staticmethod
    def _make_divisible(value: float, divisor: int) -> int:
        new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)

        # Make sure that round down does not go down by more than 10%.
        if new_value < 0.9 * value:
            new_value += divisor

        return new_value

    def forward(self, x: Tensor) -> Tensor:
        # loop through feature layers to get model output.
        return reduce(lambda y, n: self.features[n](y), list(range(0, 19)), x)


class MobileNetV2Backbone(nn.Module):
    """ MobileNetV2 Backbone"""

    def __init__(self, in_channels: int):
        super().__init__()

        self.model = MobileNetV2(in_channels, alpha=1.0, expansion=6)
        self.enc_channels = [16, 24, 32, 96, 1280]

    def forward(self, x: Tensor) -> List[Tensor]:
        x = reduce(lambda y, n: self.model.features[n](y), list(range(0, 2)), x)
        enc2x = x

        x = reduce(lambda y, n: self.model.features[n](y), list(range(2, 4)), x)
        enc4x = x

        x = reduce(lambda y, n: self.model.features[n](y), list(range(4, 7)), x)
        enc8x = x

        x = reduce(lambda y, n: self.model.features[n](y), list(range(7, 14)), x)
        enc16x = x

        x = reduce(lambda y, n: self.model.features[n](y), list(range(14, 19)), x)
        enc32x = x

        return [enc2x, enc4x, enc8x, enc16x, enc32x]
