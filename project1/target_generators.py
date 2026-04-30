"""
Target image generators.

Supports three modes:
- "image": load a real image file
- "txt_gaussians": render Gaussians defined in a txt file
- "synthetic_shapes": generate random geometric shapes (for quick testing)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from config import Config
from utils import load_rgb_image, make_synthetic_target_image, resolve_device, save_image


# ---------------------------------------------------------------------------
# txt Gaussian rendering
# ---------------------------------------------------------------------------

@dataclass
class _GaussianSpec:
    x: float
    y: float
    sigma_x: float
    sigma_y: float
    theta: float
    alpha: float
    r: float
    g: float
    b: float


def _parse_line(values: list[float], line_idx: int, txt_path: Path) -> _GaussianSpec:
    """Parse one line of a Gaussian spec file.

    Supported formats (per line):
        x  y  sigma  r  g  b                           (6 values, isotropic, opaque)
        x  y  sigma  alpha  r  g  b                    (7 values, isotropic, translucent)
        x  y  sigma_x  sigma_y  theta  r  g  b         (8 values, anisotropic, opaque)
        x  y  sigma_x  sigma_y  theta  alpha  r  g  b  (9 values, anisotropic, translucent)
    """
    n = len(values)
    if n == 6:
        x, y, sigma, r, g, b = values
        return _GaussianSpec(x, y, sigma, sigma, 0.0, 1.0, r, g, b)
    if n == 7:
        x, y, sigma, alpha, r, g, b = values
        return _GaussianSpec(x, y, sigma, sigma, 0.0, alpha, r, g, b)
    if n == 8:
        x, y, sx, sy, theta, r, g, b = values
        return _GaussianSpec(x, y, sx, sy, theta, 1.0, r, g, b)
    if n == 9:
        x, y, sx, sy, theta, alpha, r, g, b = values
        return _GaussianSpec(x, y, sx, sy, theta, alpha, r, g, b)
    raise ValueError(
        f"Line {line_idx} in {txt_path}: expected 6/7/8/9 values, got {n}."
    )


def render_txt_gaussians(
    txt_path: Path,
    image_size: int = 128,
    bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Read a txt file and render the described Gaussians into an image tensor [H, W, 3]."""
    if not txt_path.exists():
        raise FileNotFoundError(f"Gaussian txt file not found: {txt_path}")

    specs: list[_GaussianSpec] = []
    for idx, raw in enumerate(txt_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        specs.append(_parse_line([float(v) for v in line.split()], idx, txt_path))

    if not specs:
        raise ValueError(f"No valid Gaussian specs found in {txt_path}")

    device = torch.device(device) if isinstance(device, str) else device

    centers = torch.tensor([[s.x, s.y] for s in specs], dtype=torch.float32, device=device)
    scales = torch.tensor([[s.sigma_x, s.sigma_y] for s in specs], dtype=torch.float32, device=device)
    angles = torch.tensor([[s.theta] for s in specs], dtype=torch.float32, device=device)
    alphas = torch.tensor([[s.alpha] for s in specs], dtype=torch.float32, device=device)
    colors = torch.tensor([[s.r, s.g, s.b] for s in specs], dtype=torch.float32, device=device)

    coords = torch.linspace(0.0, 1.0, image_size, device=device)
    ys, xs = torch.meshgrid(coords, coords, indexing="ij")
    grid = torch.stack([xs, ys], dim=-1)  # [H, W, 2]

    diff = grid.unsqueeze(0) - centers[:, None, None, :]  # [N, H, W, 2]
    cos_t = torch.cos(angles)[:, None, None, 0]
    sin_t = torch.sin(angles)[:, None, None, 0]
    local_x = cos_t * diff[..., 0] + sin_t * diff[..., 1]
    local_y = -sin_t * diff[..., 0] + cos_t * diff[..., 1]

    sx2 = scales[:, None, None, 0] ** 2
    sy2 = scales[:, None, None, 1] ** 2
    maha = (local_x ** 2) / sx2 + (local_y ** 2) / sy2
    weights = alphas[:, None, None, 0] * torch.exp(-0.5 * maha)

    bg = torch.tensor(bg_color, dtype=torch.float32, device=device)
    weighted_rgb = torch.einsum("nhw,nc->hwc", weights, colors)
    image = bg.view(1, 1, 3) + weighted_rgb
    return image


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_target_generator(config: Config):
    """Return a generator object with a .generate(project_root, device) method."""
    name = config.target.name

    if name == "image":
        return _ImageTarget(config)
    if name == "txt_gaussians":
        return _TxtTarget(config)
    if name == "synthetic_shapes":
        return _SyntheticTarget(config)

    raise ValueError(f"Unknown target generator name: {name}")


class _ImageTarget:
    def __init__(self, config: Config) -> None:
        self.config = config

    def generate(self, project_root: Path, device: torch.device) -> torch.Tensor:
        target_path = project_root / self.config.target.image_path
        if not target_path.exists():
            raise FileNotFoundError(
                f"Target image not found: {target_path}\n"
                f"Please set config.target.image_path to a valid image path."
            )
        return load_rgb_image(target_path, image_size=self.config.target.image_size, device=device)


class _TxtTarget:
    def __init__(self, config: Config) -> None:
        self.config = config

    def generate(self, project_root: Path, device: torch.device) -> torch.Tensor:
        return render_txt_gaussians(
            txt_path=project_root / self.config.target.gaussian_txt_path,
            image_size=self.config.target.image_size,
            bg_color=self.config.render.bg_color,
            device=device,
        )


class _SyntheticTarget:
    def __init__(self, config: Config) -> None:
        self.config = config

    def generate(self, project_root: Path, device: torch.device) -> torch.Tensor:
        return make_synthetic_target_image(image_size=self.config.target.image_size, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a txt Gaussian spec to an image")
    parser.add_argument(
        "txt_path",
        nargs="?",
        default="data/txt/s1_night_cityscape.txt",
        help="Path to the txt Gaussian spec file",
    )
    parser.add_argument("--size", type=int, default=128, help="Output image size")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output image path")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    txt_path = project_root / args.txt_path
    device = resolve_device("auto")

    image = render_txt_gaussians(txt_path=txt_path, image_size=args.size, device=device)
    output_path = Path(args.output) if args.output else project_root / "outputs" / f"{txt_path.stem}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(image, output_path)

    print(f"Input:  {txt_path}")
    print(f"Size:   {args.size}x{args.size}")
    print(f"Saved:  {output_path}")


if __name__ == "__main__":
    main()
