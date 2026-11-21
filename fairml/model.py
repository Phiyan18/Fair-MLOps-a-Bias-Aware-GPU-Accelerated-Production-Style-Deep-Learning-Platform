"""PyTorch model definitions (training side only - serving uses ONNX Runtime)."""
from __future__ import annotations

import torch
from torch import nn


class DenseLayer(nn.Module):
    """y = x @ W + b  (explicit MatMul + Add).

    Written by hand instead of nn.Linear so the exported ONNX graph contains plain MatMul nodes,
    which ONNX Runtime's dynamic INT8 quantisation is guaranteed to handle.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias


class TabularMLP(nn.Module):
    """Feed-forward network for tabular data: encoder -> linear head. Outputs logits (N, 1)."""

    def __init__(self, in_features: int, hidden: tuple[int, ...] = (128, 64), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden:
            layers += [DenseLayer(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.head = DenseLayer(prev, 1)
        self.out_features = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class _GradReverse(torch.autograd.Function):
    """Identity in the forward pass, multiplies the gradient by -lambda in the backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:  # type: ignore[override]
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.lam * grad_output, None


class Adversary(nn.Module):
    """Tries to predict the protected attribute from the encoder's representation.

    Through the gradient-reversal layer the encoder is trained to make this impossible, i.e. the
    learned representation carries (almost) no information about the protected attribute
    - in-processing bias mitigation (adversarial debiasing).
    """

    def __init__(self, in_features: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(DenseLayer(in_features, hidden), nn.ReLU(), DenseLayer(hidden, 1))

    def forward(self, z: torch.Tensor, lam: float) -> torch.Tensor:
        return self.net(_GradReverse.apply(z, lam))


class ProbabilityHead(nn.Module):
    """Wraps a logit model with a sigmoid -> used for ONNX export. Argument is named `x`."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))
