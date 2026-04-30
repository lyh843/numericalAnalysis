from __future__ import annotations

from dataclasses import dataclass, field


_current_mode: str = "student"


def set_mode(mode: str) -> None:
    global _current_mode
    if mode not in ("student", "teacher"):
        raise ValueError(f"Mode must be 'student' or 'teacher', got '{mode}'")
    _current_mode = mode


def get_mode() -> str:
    return _current_mode


def is_teacher() -> bool:
    return _current_mode == "teacher"


@dataclass
class SystemConfig:
    seed: int = 42
    device: str = "auto"
    output_dir: str = "outputs"


@dataclass
class TargetConfig:
    name: str = "image"           # "image" | "txt_gaussians" | "synthetic_shapes"
    image_size: int = 128
    image_path: str = "data/real_images/r1_flamingo_128.png"
    gaussian_txt_path: str = "data/txt/s1_night_cityscape.txt"


@dataclass
class ModelConfig:
    num_gaussians: int = 1000
    use_anisotropic: bool = True
    use_alpha: bool = True


@dataclass
class RenderConfig:
    bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    eps: float = 1e-6


@dataclass
class TrainConfig:
    num_steps: int = 200
    print_every: int = 20
    save_every: int = 50
    save_video: bool = True
    video_every: int = 10


@dataclass
class LossConfig:
    name: str = "mse"


@dataclass
class ParamGroupConfig:
    """Per-parameter-group learning rate multipliers.
    effective_lr = base_lr * scale
    """
    center_lr_scale: float = 1.0
    scale_lr_scale: float = 1.0
    angle_lr_scale: float = 1.0
    alpha_lr_scale: float = 1.0
    color_lr_scale: float = 1.0


@dataclass
class SchedulerConfig:
    name: str = "constant"        # "constant" | "cosine" | "warmup_cosine" | "step_decay"


@dataclass
class OptimizerConfig:
    name: str = "torch_adam"
    lr: float = 5e-2
    param_groups: ParamGroupConfig = field(default_factory=ParamGroupConfig)


@dataclass
class InitializerConfig:
    name: str = "random"          # "random" | "grid" | "image_sample"


@dataclass
class Config:
    system: SystemConfig = field(default_factory=SystemConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    initializer: InitializerConfig = field(default_factory=InitializerConfig)
