from __future__ import annotations

from config import Config


def _build_locked_config(num_steps: int) -> Config:
    config = Config()

    # Locked fields: do not modify these in task 2.
    config.system.seed = 42
    config.target.image_size = 128
    config.model.num_gaussians = 1000
    config.render.bg_color = (0.0, 0.0, 0.0)
    config.train.num_steps = num_steps

    return config


def get_sprint_setting() -> Config:
    config = _build_locked_config(num_steps=100)

    # Editable fields: students may tune these.
    config.loss.name = "mse"
    config.initializer.name = "random"
    config.optimizer.name = "torch_adam"
    config.optimizer.lr = 5e-2
    config.scheduler.name = "constant"
    config.model.use_anisotropic = True
    config.model.use_alpha = True

    return config


def get_standard_setting() -> Config:
    config = _build_locked_config(num_steps=500)

    # Editable fields: students may tune these.
    config.loss.name = "mse"
    config.initializer.name = "random"
    config.optimizer.name = "torch_adam"
    config.optimizer.lr = 5e-2
    config.scheduler.name = "constant"
    config.model.use_anisotropic = True
    config.model.use_alpha = True

    return config
