from typing import List

from utils.modules import Scale

import torch
from torch import nn, Tensor
from torchvision import models, transforms

from functools import partial


class VGGFeatures(nn.Module):
    poolings = {'max': nn.MaxPool2d, 'average': nn.AvgPool2d, 'l2': partial(nn.LPPool2d, 2)}
    pooling_scales = {'max': 1., 'average': 2., 'l2': 0.78}

    def __init__(self, layers: List[int], pooling: str = 'max', device=torch.device("cpu")):
        super().__init__()

        self.device = device
        self.layers = sorted(set(layers))

        # Notable layers represents the layers where changes in the number of channels occur.
        self.notable_layers = [4, 9, 18, 27, 36]

        # The PyTorch pre-trained VGG-19 expects sRGB inputs in the range [0, 1] which are then
        # normalized according to this transform, unlike Simonyan et al.'s original model.
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        # The PyTorch pre-trained VGG-19 has different parameters from Simonyan et al.'s original
        # model.
        self.model = models.vgg19(pretrained=True).features[:self.layers[-1] + 1]

        # Reduces edge artifacts.
        self.model[0] = self._change_padding_mode(self.model[0], 'replicate')

        pool_scale = self.pooling_scales[pooling]
        for i, layer in enumerate(self.model):
            if pooling != 'max' and isinstance(layer, nn.MaxPool2d):
                # Changing the pooling type from max results in the scale of activations
                # changing, so rescale them. Gatys et al. (2015) do not do this.
                self.model[i] = Scale(self.poolings[pooling](2), pool_scale)

        # Load layers to the specified device.
        for i, layer in enumerate(self.model):
            self.model[i] = layer.to(self.device)

        self.model.eval()
        self.model.requires_grad_(False)

    @staticmethod
    def _change_padding_mode(conv: nn.Conv2d, padding_mode: str) -> nn.Conv2d:
        new_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                             stride=conv.stride, padding=conv.padding,
                             padding_mode=padding_mode)
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight)
            new_conv.bias.copy_(conv.bias)

        return new_conv

    def forward(self, input: Tensor, layers: List[int] = None) -> dict:
        layers = self.layers if layers is None else sorted(set(layers))
        feats = {}

        input = self.normalize(input.to(self.device))
        for layer in range(max(layers) + 1):
            input = self.model[layer](input)

            if layer in layers:
                feats[layer] = input

        return feats
