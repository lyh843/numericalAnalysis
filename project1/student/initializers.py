from __future__ import annotations

import math

import torch

from config import Config, is_teacher
from models import Gaussian2DModel, inverse_sigmoid, inverse_softplus


class RandomGaussianInitializer:
    def __init__(self, config: Config) -> None:
        self.config = config

    def initialize(self, model: Gaussian2DModel, target_image: torch.Tensor | None = None) -> None:
        del target_image
        num_gaussians = model.num_gaussians
        device = model.center_raw.device

        center_init = torch.rand(num_gaussians, 2, device=device)
        sigma_init = 0.01 + 0.01 * torch.rand(num_gaussians, 1, device=device)
        scale_init = sigma_init.repeat(1, 2)
        rotation_init = torch.zeros(num_gaussians, 2, device=device)
        rotation_init[:, 0] = 1.0
        alpha_value = 0.2 if self.config.model.use_alpha else 1.0
        alpha_init = torch.full((num_gaussians, 1), alpha_value, device=device)
        color_init = 0.5 + 0.3 * torch.randn(num_gaussians, 3, device=device)
        color_init = color_init.clamp(0.05, 0.95)

        model.set_raw_parameters(
            center_raw=inverse_sigmoid(center_init),
            scale_raw=inverse_softplus(scale_init),
            rotation_raw=rotation_init,
            alpha_raw=inverse_sigmoid(alpha_init),
            color_raw=inverse_sigmoid(color_init),
        )


class GridGaussianInitializer:
    def __init__(self, config: Config) -> None:
        self.config = config

    def initialize(self, model: Gaussian2DModel, target_image: torch.Tensor | None = None) -> None:
        raise NotImplementedError("TODO: implement grid initialization")


class ImageSampleGaussianInitializer:
    def __init__(self, config: Config) -> None:
        self.config = config

    def initialize(self, model: Gaussian2DModel, target_image: torch.Tensor | None = None) -> None:
        raise NotImplementedError("TODO: implement image_sample initialization")


def build_initializer(config: Config):
    name = config.initializer.name

    if name == "random":
        return RandomGaussianInitializer(config)
    if name == "grid":
        if is_teacher():
            from _teacher_solutions.grid_init import GridGaussianInitializer as TeacherGridGaussianInitializer
            return TeacherGridGaussianInitializer(config)
        return GridGaussianInitializer(config)
    if name == "image_sample":
        if is_teacher():
            from _teacher_solutions.image_sample_init import ImageSampleGaussianInitializer as TeacherImageSampleGaussianInitializer
            return TeacherImageSampleGaussianInitializer(config)
        return ImageSampleGaussianInitializer(config)

    raise ValueError(f"Unknown initializer name: {name}")
