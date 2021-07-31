import torch

from torch import nn, Tensor
from torch.nn import functional as F


class ScaledMSELoss(nn.Module):
    """Computes MSE scaled such that its gradient L1 norm is approximately 1.
    This differs from Gatys at al. (2015) and Johnson et al."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.register_buffer('eps', torch.tensor(eps))

    def extra_repr(self) -> str:
        return f'eps={self.eps:g}'

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        diff = input - target
        return diff.pow(2).sum() / diff.abs().sum().add(self.eps)


class ContentLoss(nn.Module):
    def __init__(self, target: Tensor, eps: float = 1e-8):
        super().__init__()
        self.register_buffer('target', target)
        self.loss = ScaledMSELoss(eps=eps)

    def forward(self, input: Tensor) -> Tensor:
        return self.loss(input, self.target)


class StyleLoss(nn.Module):
    def __init__(self, target: Tensor, eps: float = 1e-8):
        super().__init__()
        self.register_buffer('target', target)
        self.loss = ScaledMSELoss(eps=eps)

    @staticmethod
    def get_target(target: Tensor) -> Tensor:
        mat = target.flatten(-2)
        # The Gram matrix normalization differs from Gatys et al. (2015) and Johnson et al.
        return mat @ mat.transpose(-2, -1) / mat.shape[-1]

    def forward(self, input: Tensor) -> Tensor:
        return self.loss(self.get_target(input), self.target)


class TVLoss(nn.Module):
    """L2 total variation loss, as in Mahendran et al."""

    @staticmethod
    def forward(input: Tensor) -> Tensor:
        input = F.pad(input, (0, 1, 0, 1), 'replicate')

        x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
        y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]

        return (x_diff ** 2 + y_diff ** 2).mean()


class SumLoss(nn.ModuleList):
    def __init__(self, losses, verbose=False):
        super().__init__(losses)
        self.verbose = verbose

    def forward(self, *args, **kwargs) -> Tensor:
        losses = [loss(*args, **kwargs) for loss in self]

        if self.verbose:
            for i, loss in enumerate(losses):
                print(f'({i}): {loss.item():g}')

        return sum(loss.to(losses[-1].device) for loss in losses)
