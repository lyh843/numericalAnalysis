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
        # raise NotImplementedError("TODO: implement grid initialization")
        del target_image
        num_gaussians = model.num_gaussians
        device = model.center_raw.device
        
        cols = math.ceil(math.sqrt(num_gaussians))
        rows = math.ceil(num_gaussians / cols)
        
        xs = torch.linspace(0.0, 1.0, cols + 2, device=device)[1:-1]
        ys = torch.linspace(0.0, 1.0, rows + 2, device=device)[1:-1]
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        center_init = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)[:num_gaussians]
        
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
        

class ImageSampleGaussianInitializer:
    def __init__(self, config: Config) -> None:
        self.config = config

    def initialize(self, model: Gaussian2DModel, target_image: torch.Tensor | None = None) -> None:
        # raise NotImplementedError("TODO: implement image_sample initialization")
        num_gaussians = model.num_gaussians
        device = model.center_raw.device

        center_init = torch.rand(num_gaussians, 2, device=device)
        sigma_init = 0.01 + 0.01 * torch.rand(num_gaussians, 1, device=device)
        scale_init = sigma_init.repeat(1, 2)
        rotation_init = torch.zeros(num_gaussians, 2, device=device)
        rotation_init[:, 0] = 1.0
        alpha_value = 0.2 if self.config.model.use_alpha else 1.0
        alpha_init = torch.full((num_gaussians, 1), alpha_value, device=device)
        
        h, w, _ = target_image.shape
        px = torch.clamp((center_init[:, 0] * (w - 1)).round().long(), 0, w - 1)
        py = torch.clamp((center_init[:, 1] * (h - 1)).round().long(), 0, h - 1)
        color_init = target_image[py, px].clamp(0.05, 0.95)

        model.set_raw_parameters(
            center_raw=inverse_sigmoid(center_init),
            scale_raw=inverse_softplus(scale_init),
            rotation_raw=rotation_init,
            alpha_raw=inverse_sigmoid(alpha_init),
            color_raw=inverse_sigmoid(color_init),
        )
        
class ImportanceGaussianInitializer:
    def __init__(self, config: Config) -> None:
        self.config = config

    def initialize(self, model: Gaussian2DModel, target_image: torch.Tensor | None = None) -> None:
        num_gaussians = model.num_gaussians
        device = model.center_raw.device

        image = target_image.to(device=device, dtype=torch.float32)
        h, w, _ = image.shape

        gray = (
            0.299 * image[..., 0]
            + 0.587 * image[..., 1]
            + 0.114 * image[..., 2]
        ).unsqueeze(0).unsqueeze(0)

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=device,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=device,
        ).view(1, 1, 3, 3)

        grad_x = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
        grad_y = torch.nn.functional.conv2d(gray, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)

        local_mean = torch.nn.functional.avg_pool2d(gray, kernel_size=5, stride=1, padding=2)
        local_mean_sq = torch.nn.functional.avg_pool2d(gray.square(), kernel_size=5, stride=1, padding=2)
        local_var = (local_mean_sq - local_mean.square()).clamp_min(0.0)

        edge_strength = torch.nn.functional.avg_pool2d(grad_mag, kernel_size=3, stride=1, padding=1)

        importance = grad_mag + 0.5 * edge_strength + 0.5 * local_var
        importance = importance.squeeze(0).squeeze(0)
        importance = importance / importance.mean().clamp_min(1e-8)
        prob = (importance + 1e-4).reshape(-1)
        prob = prob / prob.sum().clamp_min(1e-8)

        sampled_idx = torch.multinomial(prob, num_gaussians, replacement=True)
        py = sampled_idx // w
        px = sampled_idx % w

        jitter_x = (torch.rand(num_gaussians, device=device) - 0.5) / max(w, 1)
        jitter_y = (torch.rand(num_gaussians, device=device) - 0.5) / max(h, 1)
        center_x = (px.float() + 0.5) / max(w, 1) + jitter_x
        center_y = (py.float() + 0.5) / max(h, 1) + jitter_y
        center_init = torch.stack([center_x, center_y], dim=-1).clamp(1e-4, 1.0 - 1e-4)

        # if num_gaussians <= 1:
        #     sigma_init = torch.full((num_gaussians, 1), 0.02, device=device)
        # else:
        pairwise_dist = torch.cdist(center_init, center_init)
        pairwise_dist.fill_diagonal_(float("inf"))

        knn_k = min(8, num_gaussians - 1)
        knn_dist, _ = torch.topk(pairwise_dist, k=knn_k, dim=1, largest=False)
        local_spacing = knn_dist.mean(dim=1, keepdim=True)

        sigma_init = 0.01 + 0.01 * torch.rand(num_gaussians, 1, device=device)

        scale_init = sigma_init.repeat(1, 2)
        rotation_init = torch.zeros(num_gaussians, 2, device=device)
        rotation_init[:, 0] = 1.0
        alpha_value = 0.2 if self.config.model.use_alpha else 1.0
        alpha_init = torch.full((num_gaussians, 1), alpha_value, device=device)
        
        image = target_image.to(device=device, dtype=torch.float32)
        h, w, _ = image.shape
        px = torch.clamp((center_init[:, 0] * (w - 1)).round().long(), 0, w - 1)
        py = torch.clamp((center_init[:, 1] * (h - 1)).round().long(), 0, h - 1)
        color_init = image[py, px].clamp(0.05, 0.95)

        model.set_raw_parameters(
            center_raw=inverse_sigmoid(center_init),
            scale_raw=inverse_softplus(scale_init),
            rotation_raw=rotation_init,
            alpha_raw=inverse_sigmoid(alpha_init),
            color_raw=inverse_sigmoid(color_init),
        )

class ImportanceGaussianInitializerForTaskA:
    def __init__(self, config: Config) -> None:
        self.config = config

    def initialize(self, model: Gaussian2DModel, target_image: torch.Tensor | None = None) -> None:
        num_gaussians = model.num_gaussians
        device = model.center_raw.device

        image = target_image.to(device=device, dtype=torch.float32)
        h, w, _ = image.shape

        gray = (
            0.299 * image[..., 0]
            + 0.587 * image[..., 1]
            + 0.114 * image[..., 2]
        ).unsqueeze(0).unsqueeze(0)

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=device,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=device,
        ).view(1, 1, 3, 3)

        grad_x = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
        grad_y = torch.nn.functional.conv2d(gray, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)

        local_mean = torch.nn.functional.avg_pool2d(gray, kernel_size=5, stride=1, padding=2)
        local_mean_sq = torch.nn.functional.avg_pool2d(gray.square(), kernel_size=5, stride=1, padding=2)
        local_var = (local_mean_sq - local_mean.square()).clamp_min(0.0)

        edge_strength = torch.nn.functional.avg_pool2d(grad_mag, kernel_size=3, stride=1, padding=1)

        importance = grad_mag + 0.5 * edge_strength + 0.5 * local_var
        importance = importance.squeeze(0).squeeze(0)
        importance = importance / importance.mean().clamp_min(1e-8)
        prob = (importance + 1e-4).reshape(-1)
        prob = prob / prob.sum().clamp_min(1e-8)

        sampled_idx = torch.multinomial(prob, num_gaussians, replacement=True)
        py = sampled_idx // w
        px = sampled_idx % w

        jitter_x = (torch.rand(num_gaussians, device=device) - 0.5) / max(w, 1)
        jitter_y = (torch.rand(num_gaussians, device=device) - 0.5) / max(h, 1)
        center_x = (px.float() + 0.5) / max(w, 1) + jitter_x
        center_y = (py.float() + 0.5) / max(h, 1) + jitter_y
        center_init = torch.stack([center_x, center_y], dim=-1).clamp(1e-4, 1.0 - 1e-4)


        pairwise_dist = torch.cdist(center_init, center_init)
        pairwise_dist.fill_diagonal_(float("inf"))

        knn_k = min(8, num_gaussians - 1)
        knn_dist, _ = torch.topk(pairwise_dist, k=knn_k, dim=1, largest=False)
        local_spacing = knn_dist.mean(dim=1, keepdim=True)

        # Denser sampled regions get smaller Gaussians, while sparse regions
        # start a bit wider to cover area earlier in optimization.
        sigma_init = (0.35 * local_spacing).clamp(0.005, 0.05)

        scale_init = sigma_init.repeat(1, 2)
        rotation_init = torch.zeros(num_gaussians, 2, device=device)
        rotation_init[:, 0] = 1.0
        alpha_value = 0.2 if self.config.model.use_alpha else 1.0
        alpha_init = torch.full((num_gaussians, 1), alpha_value, device=device)
        
        image = target_image.to(device=device, dtype=torch.float32)
        h, w, _ = image.shape
        px = torch.clamp((center_init[:, 0] * (w - 1)).round().long(), 0, w - 1)
        py = torch.clamp((center_init[:, 1] * (h - 1)).round().long(), 0, h - 1)
        color_init = image[py, px].clamp(0.05, 0.95)

        model.set_raw_parameters(
            center_raw=inverse_sigmoid(center_init),
            scale_raw=inverse_softplus(scale_init),
            rotation_raw=rotation_init,
            alpha_raw=inverse_sigmoid(alpha_init),
            color_raw=inverse_sigmoid(color_init),
        )


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
    if name == "importance":
        return ImportanceGaussianInitializer(config)
    if name == "importanceTaskA":
        return ImportanceGaussianInitializerForTaskA(config)

    raise ValueError(f"Unknown initializer name: {name}")
