from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from config import Config, set_mode
from models import Gaussian2DModel
from renderer import GaussianRenderer
from student.initializers import build_initializer
from student.losses import build_loss
from student.optimizers import build_optimizer
from student.schedulers import build_scheduler
from target_generators import build_target_generator
from utils import (
    MetricTracker,
    build_animation_from_frames,
    ensure_dir,
    plot_loss_curve,
    resolve_device,
    save_image,
    save_image_panel,
    save_training_frame,
    set_seed,
)


@dataclass
class EvalResult:
    mse: float
    mae: float
    psnr: float


def evaluate_prediction(prediction: torch.Tensor, target: torch.Tensor) -> EvalResult:
    mse = torch.mean((prediction - target) ** 2).item()
    mae = torch.mean(torch.abs(prediction - target)).item()
    psnr = -10.0 * torch.log10(torch.tensor(mse + 1e-12)).item()
    return EvalResult(mse=mse, mae=mae, psnr=psnr)


def save_evaluation_report(
    result: EvalResult,
    path: str | Path,
    optimizer_name: str,
    init_name: str,
    num_steps: int,
    image_size: int,
) -> None:
    lines = [
        "2DGS Evaluation Summary",
        f"optimizer: {optimizer_name}",
        f"initializer: {init_name}",
        f"num_steps: {num_steps}",
        f"image_size: {image_size}",
        f"mse: {result.mse:.8f}",
        f"mae: {result.mae:.8f}",
        f"psnr: {result.psnr:.4f} dB",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_comparison_visual(target: torch.Tensor, prediction: torch.Tensor, path: str | Path) -> None:
    error = torch.abs(prediction - target)
    save_image_panel(
        images=[target, prediction, error],
        titles=["Target", "Prediction", "Absolute Error"],
        path=path,
    )


def _apply_scheduler(optimizer, lr_scale: float) -> None:
    """Update learning rates in all parameter groups by scaling from base_lr."""
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * lr_scale


def train(config: Config) -> None:
    """
    Run the default 2DGS training loop.

    This function intentionally keeps the whole pipeline visible:
    target -> model -> renderer -> loss -> optimizer -> scheduler.
    """
    set_seed(config.system.seed)

    device = resolve_device(config.system.device)
    project_root = Path(__file__).resolve().parent
    output_dir = ensure_dir(project_root / config.system.output_dir)
    frame_paths: list[Path] = []
    frames_dir: Path | None = None
    if config.train.save_video:
        frames_dir = ensure_dir(output_dir / "video_frames")

    target_generator = build_target_generator(config)
    target = target_generator.generate(project_root=project_root, device=device)
    save_image(target, output_dir / "target.png")

    model = Gaussian2DModel(num_gaussians=config.model.num_gaussians).to(device)
    initializer = build_initializer(config)
    initializer.initialize(model, target_image=target)

    renderer = GaussianRenderer(
        image_size=config.target.image_size,
        bg_color=config.render.bg_color,
        use_anisotropic=config.model.use_anisotropic,
        use_alpha=config.model.use_alpha,
    )
    optimizer = build_optimizer(model=model, config=config.optimizer)
    loss_fn = build_loss(config.loss)
    scheduler = build_scheduler(config.scheduler)
    metric_tracker = MetricTracker()

    losses: list[float] = []

    print(f"Device: {device}")
    print(f"Image size: {config.target.image_size}x{config.target.image_size}")
    print(f"Number of Gaussians: {config.model.num_gaussians}")
    print(f"Use anisotropic: {config.model.use_anisotropic}")
    print(f"Use alpha: {config.model.use_alpha}")
    print(f"Target generator: {config.target.name}")
    print(f"Initializer: {config.initializer.name}")
    print(f"Optimizer: {config.optimizer.name}")
    print(f"Scheduler: {config.scheduler.name}")
    print(f"Loss: {config.loss.name}")
    print(f"Save video: {config.train.save_video}")

    for step in range(1, config.train.num_steps + 1):
        lr_scale = scheduler(step, config.train.num_steps)
        _apply_scheduler(optimizer, lr_scale)

        optimizer.zero_grad()

        render_params = model.get_render_params()
        prediction = renderer.render(render_params)

        loss = loss_fn(prediction, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        metric_tracker.log(
            step=step,
            metrics={"total_loss": loss.item()},
        )

        if step == 1 or step % config.train.print_every == 0 or step == config.train.num_steps:
            print(f"step={step:04d}  loss={loss.item():.6f}")

        if step % config.train.save_every == 0 or step == config.train.num_steps:
            save_image(prediction, output_dir / f"recon_step_{step:04d}.png")

        if config.train.save_video and (
            step == 1
            or step % config.train.video_every == 0
            or step == config.train.num_steps
        ):
            assert frames_dir is not None
            frame_path = frames_dir / f"frame_{step:04d}.png"
            save_training_frame(
                target=target,
                prediction=prediction,
                path=frame_path,
                step=step,
            )
            frame_paths.append(frame_path)

    save_image(prediction, output_dir / "final_reconstruction.png")
    plot_loss_curve(losses, output_dir / "loss_curve.png")
    metric_tracker.save_json(output_dir / "metrics.json")
    metric_tracker.plot(output_dir / "metric_curves.png")

    eval_result = evaluate_prediction(prediction=prediction, target=target)
    save_evaluation_report(
        result=eval_result,
        path=output_dir / "evaluation.txt",
        optimizer_name=config.optimizer.name,
        init_name=config.initializer.name,
        num_steps=config.train.num_steps,
        image_size=config.target.image_size,
    )
    save_comparison_visual(
        target=target,
        prediction=prediction,
        path=output_dir / "comparison.png",
    )

    if config.train.save_video:
        animation_path = build_animation_from_frames(
            frame_paths=frame_paths,
            output_path=output_dir / "optimization.mp4",
        )
        print(f"Saved optimization animation to: {animation_path}")

    print(
        "Final evaluation: "
        f"MSE={eval_result.mse:.8f}, "
        f"MAE={eval_result.mae:.8f}, "
        f"PSNR={eval_result.psnr:.4f} dB"
    )

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Gaussian Splatting image fitting")
    parser.add_argument("--mode", choices=["student", "teacher"], default="student")
    args = parser.parse_args()
    set_mode(args.mode)
    train(Config())
