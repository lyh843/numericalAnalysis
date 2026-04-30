from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ParamGroupConfig


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(1e-4, 1.0 - 1e-4)
    return torch.log(x / (1.0 - x))


def inverse_softplus(y: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(y))


@dataclass
class GaussianRenderParams:
    centers: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    alphas: torch.Tensor
    colors: torch.Tensor


class Gaussian2DModel(nn.Module):
    """
    Minimal Gaussian parameter container.

    This class is intentionally small:
    - it stores raw trainable parameters
    - it applies simple constraints
    - it returns the current render-ready values

    Initialization is handled elsewhere so students can swap strategies easily.
    """

    def __init__(self, num_gaussians: int = 32) -> None:
        super().__init__()
        self.num_gaussians = num_gaussians
        default_rotation = torch.zeros(num_gaussians, 2)
        default_rotation[:, 0] = 1.0

        self.center_raw = nn.Parameter(torch.zeros(num_gaussians, 2))
        self.scale_raw = nn.Parameter(torch.zeros(num_gaussians, 2))
        self.rotation_raw = nn.Parameter(default_rotation)
        self.alpha_raw = nn.Parameter(torch.zeros(num_gaussians, 1))
        self.color_raw = nn.Parameter(torch.zeros(num_gaussians, 3))

    def get_render_params(self) -> GaussianRenderParams:
        centers = torch.sigmoid(self.center_raw)
        scales = F.softplus(self.scale_raw) + 1e-4
        rotations = F.normalize(self.rotation_raw, dim=-1, eps=1e-8)
        alphas = torch.sigmoid(self.alpha_raw)
        colors = torch.sigmoid(self.color_raw)
        return GaussianRenderParams(
            centers=centers,
            scales=scales,
            rotations=rotations,
            alphas=alphas,
            colors=colors,
        )

    def get_param_groups(self, base_lr: float, group_config: ParamGroupConfig) -> list[dict]:
        """
        Build parameter groups with per-group learning rates.

        Each group contains one parameter tensor and its effective learning rate.
        This allows the optimizer to use different learning rates for
        position, scale, rotation, opacity, and color.
        """
        return [
            {"name": "center", "params": [self.center_raw], "lr": base_lr * group_config.center_lr_scale, "base_lr": base_lr * group_config.center_lr_scale},
            {"name": "scale",  "params": [self.scale_raw],  "lr": base_lr * group_config.scale_lr_scale,  "base_lr": base_lr * group_config.scale_lr_scale},
            {"name": "rotation",  "params": [self.rotation_raw],  "lr": base_lr * group_config.angle_lr_scale,  "base_lr": base_lr * group_config.angle_lr_scale},
            {"name": "alpha",  "params": [self.alpha_raw],  "lr": base_lr * group_config.alpha_lr_scale,  "base_lr": base_lr * group_config.alpha_lr_scale},
            {"name": "color",  "params": [self.color_raw],  "lr": base_lr * group_config.color_lr_scale,  "base_lr": base_lr * group_config.color_lr_scale},
        ]

    def set_raw_parameters(
        self,
        center_raw: torch.Tensor,
        scale_raw: torch.Tensor,
        rotation_raw: torch.Tensor,
        alpha_raw: torch.Tensor,
        color_raw: torch.Tensor,
    ) -> None:
        """
        Helper used by initialization modules.
        """
        with torch.no_grad():
            self.center_raw.copy_(center_raw)
            self.scale_raw.copy_(scale_raw)
            self.rotation_raw.copy_(rotation_raw)
            self.alpha_raw.copy_(alpha_raw)
            self.color_raw.copy_(color_raw)
