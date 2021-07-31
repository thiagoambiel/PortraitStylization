import warnings
from typing import Tuple, Any, List

import copy
from dataclasses import dataclass

import torch
from torch import nn, Tensor
from torch.nn import functional as F


@dataclass
class Model:
    tag: str
    model: nn.Module


class Layers:
    def __init__(self, tag: str, layers: List[int], hidden_layers: List[int] = ()):
        self.tag = tag
        self.layers = sorted(set(layers))
        self.hidden_layers = hidden_layers

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self) -> str:
        for layer in self.layers:
            if layer not in self.hidden_layers:
                yield f'{self.tag}_{layer}'

    def load(self, layers):
        for layer in layers:
            if layer not in self.layers:
                self.layers.append(layer)

    def append(self, layer):
        if layer not in self.layers:
            self.layers.append(layer)


class MergeModels(nn.Module):
    def __init__(self, models: List[Model]):
        super().__init__()
        self.models = models

    def get_model(self, tag: str) -> nn.Module:
        for model_data in self.models:
            if model_data.tag == tag:
                return model_data.model

    def get_features(self, input: Tensor, layers: Layers) -> dict:
        for model_data in self.models:
            if model_data.tag == layers.tag:
                features = model_data.model(input, layers=layers.layers)

                for layer in copy.copy(features):
                    features[f'{layers.tag}_{layer}'] = features.pop(layer)

                return features

        raise ValueError("'layers.tag' must be the same as the tag of one initialized model.")

    def forward(self, input: Tensor, models_layers: List[Layers]) -> dict:
        features = {'input': input}

        for layers in models_layers:
            feats = self.get_features(input, layers)
            features.update(feats)

        return features


class Scale(nn.Module):
    def __init__(self, module: nn.Module, scale: float):
        super().__init__()
        self.module = module
        self.register_buffer('scale', torch.tensor(scale))

    def extra_repr(self) -> str:
        return f'(scale): {self.scale.item():g}'

    def forward(self, *args, **kwargs) -> Any:
        return self.module(*args, **kwargs) * self.scale


class LayerApply(nn.Module):
    def __init__(self, module: nn.Module, layer: Any):
        super().__init__()
        self.module = module
        self.layer = layer

    def extra_repr(self) -> str:
        return f'(layer): {self.layer!r}'

    def forward(self, input) -> Any:
        return self.module(input[self.layer])


class EMA(nn.Module):
    """A bias-corrected exponential moving average, as in Kingma et al. (Adam)."""

    def __init__(self, input: Tensor, decay: float):
        super().__init__()
        self.register_buffer('value', torch.zeros_like(input))
        self.register_buffer('decay', torch.tensor(decay))
        self.register_buffer('accum', torch.tensor(1.))
        self.update(input)

    def get(self) -> Tensor:
        return self.value / (1 - self.accum)

    def update(self, input: Tensor):
        self.accum *= self.decay
        self.value *= self.decay
        self.value += (1 - self.decay) * input


def size_to_fit(size, max_dim: int, scale_up: bool = False) -> Tuple[int, int]:
    w, h = size

    if not scale_up and max(h, w) <= max_dim:
        return w, h

    new_w, new_h = max_dim, max_dim
    if h > w:
        new_w = round(max_dim * w / h)
    else:
        new_h = round(max_dim * h / w)

    return new_w, new_h


def gen_scales(start: int, end: int) -> List[Any]:
    scale = end
    scales = set()

    i = 0
    while scale >= start:
        scales.add(scale)
        scale = round(end / pow(2, i / 2))
        i += 1

    return sorted(scales)


def interpolate(*args, **kwargs) -> Tensor:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return F.interpolate(*args, **kwargs)


def scale_adam(state: dict, shape: Tuple[int, int]) -> dict:
    """Prepares a state dict to warm-start the Adam optimizer at a new scale."""
    state = copy.deepcopy(state)

    for group in state['state'].values():
        group['exp_avg'] = interpolate(group['exp_avg'], shape, mode='bicubic')
        group['exp_avg_sq'] = interpolate(group['exp_avg_sq'], shape, mode='bilinear').relu_()

        if 'max_exp_avg_sq' in group:
            group['max_exp_avg_sq'] = interpolate(group['max_exp_avg_sq'], shape, mode='bilinear').relu_()

    return state


def scale_boundaries(img_h: int, img_w: int, ref_size: int = 512) -> Tuple[int, int]:
    img_rh = img_h
    img_rw = img_w

    if max(img_h, img_w) < ref_size or min(img_h, img_w) > ref_size:

        if img_w >= img_h:
            img_rh = ref_size
            img_rw = int(img_w / img_h * ref_size)

        elif img_w < img_h:
            img_rw = ref_size
            img_rh = int(img_h / img_w * ref_size)

    img_rw = img_rw - img_rw % 32
    img_rh = img_rh - img_rh % 32

    return img_rh, img_rw


def get_model_min_size(layers: List[int], notable_layers: List[int]) -> int:
    last_layer = max(layers)
    min_size = 1

    for layer in notable_layers:
        if last_layer < layer:
            break
        min_size *= 2

    return min_size
