from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config, set_mode
from models import Gaussian2DModel
from renderer import GaussianRenderer
from student.initializers import build_initializer
from student.losses import build_loss
from student.optimizers import build_optimizer
from student.schedulers import build_scheduler
from target_generators import build_target_generator
from train import evaluate_prediction
from utils import ensure_dir, resolve_device, save_image, set_seed


TEST_IMAGES: list[dict[str, str]] = [
    {"name": "R1_flamingo", "target_name": "image", "path": "data/real_images/r1_flamingo_128.png"},
    {"name": "R2_starry_night", "target_name": "image", "path": "data/real_images/r2_starry_night_128.png"},
    {"name": "R3_parkour", "target_name": "image", "path": "data/real_images/r3_parkour_128.png"},
    {"name": "S1_night_cityscape", "target_name": "txt_gaussians", "path": "data/txt/s1_night_cityscape.txt"},
    {"name": "S2_mandala", "target_name": "txt_gaussians", "path": "data/txt/s2_mandala.txt"},
    {"name": "S3_coral_reef", "target_name": "txt_gaussians", "path": "data/txt/s3_coral_reef.txt"},
]


def load_settings_module(path: str) -> ModuleType:
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Assignment 2 settings file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("assignment2_settings", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import settings from: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_locked_fields(config: Config, num_steps: int) -> None:
    errors: list[str] = []
    if config.system.seed != 42:
        errors.append(f"seed must be 42, got {config.system.seed}")
    if config.target.image_size != 128:
        errors.append(f"image_size must be 128, got {config.target.image_size}")
    if config.model.num_gaussians != 1000:
        errors.append(f"num_gaussians must be 1000, got {config.model.num_gaussians}")
    if config.render.bg_color != (0.0, 0.0, 0.0):
        errors.append(f"bg_color must be (0, 0, 0), got {config.render.bg_color}")
    if config.train.num_steps != num_steps:
        errors.append(f"num_steps must be {num_steps}, got {config.train.num_steps}")

    if errors:
        raise ValueError("Locked task-2 fields were modified:\n" + "\n".join(errors))


def run_single_case(config: Config, test_image: dict[str, str], output_dir: Path) -> float:
    set_seed(config.system.seed)
    device = resolve_device(config.system.device)

    config.train.save_video = False
    config.system.output_dir = str(output_dir)
    config.target.name = test_image["target_name"]
    if test_image["target_name"] == "image":
        config.target.image_path = test_image["path"]
    else:
        config.target.gaussian_txt_path = test_image["path"]

    target_generator = build_target_generator(config)
    target = target_generator.generate(project_root=PROJECT_ROOT, device=device)

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

    for step in range(1, config.train.num_steps + 1):
        lr_scale = scheduler(step, config.train.num_steps)
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * lr_scale

        optimizer.zero_grad()
        prediction = renderer.render(model.get_render_params())
        loss = loss_fn(prediction, target)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        prediction = renderer.render(model.get_render_params())

    result = evaluate_prediction(prediction=prediction, target=target)

    out = ensure_dir(output_dir)
    save_image(target, out / "target.png")
    save_image(prediction, out / "prediction.png")
    return result.psnr


def run_track(
    name: str,
    get_config_fn,
    num_steps: int,
    output_dir: Path,
    limit: int | None,
) -> None:
    test_images = TEST_IMAGES if limit is None else TEST_IMAGES[:limit]
    psnrs: list[float] = []

    print(f"\n=== {name} ({num_steps} steps) ===")
    for test_image in test_images:
        config = get_config_fn()
        validate_locked_fields(config, num_steps=num_steps)
        case_output = output_dir / name.lower() / test_image["name"]
        psnr = run_single_case(config, test_image, case_output)
        psnrs.append(psnr)
        print(f"{test_image['name']:24s}  PSNR = {psnr:.4f} dB")

    avg_psnr = sum(psnrs) / len(psnrs)
    summary_path = output_dir / f"{name.lower()}_summary.txt"
    summary_lines = [f"{name} average PSNR: {avg_psnr:.4f} dB"]
    for test_image, psnr in zip(test_images, psnrs):
        summary_lines.append(f"{test_image['name']}: {psnr:.4f} dB")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"{'AVERAGE':24s}  PSNR = {avg_psnr:.4f} dB")
    print(f"Saved summary to: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local task-2 self-check")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/assignment2_settings.py",
        help="Path to task-2 settings file.",
    )
    parser.add_argument(
        "--track",
        choices=["sprint", "standard", "both"],
        default="both",
        help="Which task-2 setting to run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N test images for a quick smoke test.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs_assignment2",
        help="Output directory for local task-2 runs.",
    )
    parser.add_argument(
        "--mode",
        choices=["student", "teacher"],
        default="student",
        help="student: use student implementations; teacher: use reference solutions.",
    )
    args = parser.parse_args()

    set_mode(args.mode)
    module = load_settings_module(args.config)
    output_dir = ensure_dir(PROJECT_ROOT / args.output)

    if args.track in {"sprint", "both"}:
        if not hasattr(module, "get_sprint_setting"):
            raise AttributeError("assignment2_settings.py must define get_sprint_setting()")
        run_track("Task2A", module.get_sprint_setting, 100, output_dir, args.limit)

    if args.track in {"standard", "both"}:
        if not hasattr(module, "get_standard_setting"):
            raise AttributeError("assignment2_settings.py must define get_standard_setting()")
        run_track("Task2B", module.get_standard_setting, 500, output_dir, args.limit)


if __name__ == "__main__":
    main()
