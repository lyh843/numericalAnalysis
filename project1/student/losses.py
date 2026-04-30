from __future__ import annotations

import torch

from config import LossConfig, is_teacher


def mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = prediction - target
    return torch.mean(diff * diff)


def l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("TODO: implement l1_loss")


def charbonnier_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    raise NotImplementedError("TODO: implement charbonnier_loss")


def mse_l1_loss(prediction: torch.Tensor, target: torch.Tensor, l1_weight: float = 0.2) -> torch.Tensor:
    raise NotImplementedError("TODO: implement mse_l1_loss")


def mse_edge_loss(prediction: torch.Tensor, target: torch.Tensor, edge_weight: float = 0.1) -> torch.Tensor:
    raise NotImplementedError("TODO: implement mse_edge_loss")


def build_loss(config: LossConfig):
    name = config.name

    if name == "mse":
        return mse_loss

    if is_teacher():
        from _teacher_solutions import losses as ref

        if name == "l1":
            return ref.l1_loss
        if name == "charbonnier":
            return lambda p, t: ref.charbonnier_loss(p, t, eps=1e-3)
        if name == "mse_l1":
            return lambda p, t: ref.mse_l1_loss(p, t, l1_weight=0.2)
        if name == "mse_edge":
            return lambda p, t: ref.mse_edge_loss(p, t, edge_weight=0.1)

    if name == "l1":
        return l1_loss
    if name == "charbonnier":
        return lambda prediction, target: charbonnier_loss(prediction, target, eps=1e-3)
    if name == "mse_l1":
        return lambda prediction, target: mse_l1_loss(prediction, target, l1_weight=0.2)
    if name == "mse_edge":
        return lambda prediction, target: mse_edge_loss(prediction, target, edge_weight=0.1)

    raise ValueError(f"Unknown loss name: {name}")
