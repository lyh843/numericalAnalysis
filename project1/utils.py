from __future__ import annotations

import json
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


class MetricTracker:
    """
    Minimal metric recorder for training curves.
    """

    def __init__(self) -> None:
        self.records: list[dict[str, float]] = []

    def log(self, step: int, metrics: dict[str, float]) -> None:
        record: dict[str, float] = {"step": float(step)}
        record.update({key: float(value) for key, value in metrics.items()})
        self.records.append(record)

    def metric_names(self) -> list[str]:
        names: set[str] = set()
        for record in self.records:
            names.update(record.keys())
        names.discard("step")
        return sorted(names)

    def get_series(self, name: str) -> list[float]:
        return [float(record.get(name, float("nan"))) for record in self.records]

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.records, indent=2), encoding="utf-8")

    def plot(self, path: str | Path, title: str = "Training Metrics") -> None:
        metric_names = self.metric_names()
        if not self.records or not metric_names:
            return

        steps = self.get_series("step")
        plt.figure(figsize=(7, 4))
        for name in metric_names:
            plt.plot(steps, self.get_series(name), linewidth=2, label=name)
        plt.xlabel("Training step")
        plt.ylabel("Value")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        if len(metric_names) > 1:
            plt.legend()
        plt.tight_layout()
        plt.savefig(Path(path), dpi=150)
        plt.close()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device_name: str) -> torch.device:
    """
    Use CUDA when available, otherwise fall back to CPU.
    """
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is not available in this environment. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(device_name)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_image(image: torch.Tensor, path: str | Path) -> None:
    image_np = image.detach().cpu().clamp(0.0, 1.0).numpy()
    plt.imsave(Path(path), image_np)


def plot_loss_curve(losses: list[float], path: str | Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(losses, linewidth=2)
    plt.xlabel("Training step")
    plt.ylabel("MSE loss")
    plt.title("2DGS Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(path), dpi=150)
    plt.close()


def load_rgb_image(path: str | Path, image_size: int, device: torch.device) -> torch.Tensor:
    """
    Load an RGB image, center-crop it to a square, and resize it.
    """
    image = Image.open(path).convert("RGB")
    width, height = image.size
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    image = image.crop((left, top, left + crop_size, top + crop_size))
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    image_np = np.array(image).astype("float32") / 255.0
    return torch.from_numpy(image_np).to(device)


def save_center_cropped_image(
    src_path: str | Path,
    dst_path: str | Path,
    image_size: int,
) -> Path:
    """
    Create a deterministic square crop for the default target image.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(src_path).convert("RGB")
    width, height = image.size
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    image = image.crop((left, top, left + crop_size, top + crop_size))
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    image.save(dst_path)
    return dst_path


def save_image_panel(
    images: list[torch.Tensor],
    titles: list[str],
    path: str | Path,
) -> None:
    """
    Save several images in one row so students can compare results quickly.
    """
    num_images = len(images)
    plt.figure(figsize=(4 * num_images, 4))
    for idx, (image, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(1, num_images, idx)
        plt.imshow(image.detach().cpu().clamp(0.0, 1.0).numpy())
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(Path(path), dpi=150)
    plt.close()


def save_training_frame(
    target: torch.Tensor,
    prediction: torch.Tensor,
    path: str | Path,
    step: int,
) -> None:
    """
    Save one visualization frame for an optimization animation.
    """
    error = torch.abs(prediction - target)
    save_image_panel(
        images=[target, prediction, error],
        titles=[
            "Target",
            f"Prediction (step {step})",
            "Absolute Error",
        ],
        path=path,
    )


def build_animation_from_frames(
    frame_paths: list[str | Path],
    output_path: str | Path,
    fps: int = 8,
) -> Path:
    """
    Build an animation from saved frame images.

    Supported outputs:
    - `.gif` via Pillow
    - `.mp4` via imageio if it is available

    If `.mp4` export is requested but imageio is not available, this falls back
    to a gif with the same file stem.
    """
    if not frame_paths:
        raise ValueError("No frame paths were provided for animation export.")

    output_path = Path(output_path)
    frames = [Image.open(frame_path).convert("RGB") for frame_path in frame_paths]

    if output_path.suffix.lower() == ".gif":
        duration_ms = int(1000 / max(1, fps))
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
        for frame in frames:
            frame.close()
        return output_path

    if output_path.suffix.lower() == ".mp4":
        try:
            import imageio.v2 as imageio
        except ImportError:
            fallback_path = output_path.with_suffix(".gif")
            duration_ms = int(1000 / max(1, fps))
            frames[0].save(
                fallback_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0,
            )
            for frame in frames:
                frame.close()
            return fallback_path

        with imageio.get_writer(output_path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(np.array(frame))

        for frame in frames:
            frame.close()
        return output_path

    for frame in frames:
        frame.close()
    raise ValueError(f"Unsupported animation output format: {output_path.suffix}")


def make_synthetic_target_image(image_size: int, device: torch.device) -> torch.Tensor:
    """
    Create a simple RGB target image with a few geometric shapes.

    This keeps the starter code self-contained and avoids external downloads.
    """
    coords = torch.linspace(0.0, 1.0, image_size, device=device)
    ys, xs = torch.meshgrid(coords, coords, indexing="ij")

    image = torch.zeros(image_size, image_size, 3, device=device)

    red_circle = ((xs - 0.28) ** 2 + (ys - 0.30) ** 2) <= 0.14**2
    green_rect = (xs >= 0.55) & (xs <= 0.82) & (ys >= 0.18) & (ys <= 0.42)
    blue_circle = ((xs - 0.38) ** 2 + (ys - 0.74) ** 2) <= 0.16**2
    yellow_band = (ys >= 0.62) & (ys <= 0.78) & (xs >= 0.60) & (xs <= 0.88)

    image[red_circle] = torch.tensor([0.95, 0.20, 0.18], device=device)
    image[green_rect] = torch.tensor([0.18, 0.82, 0.28], device=device)
    image[blue_circle] = torch.tensor([0.18, 0.35, 0.95], device=device)
    image[yellow_band] = torch.tensor([0.96, 0.88, 0.20], device=device)

    return image.clamp(0.0, 1.0)
