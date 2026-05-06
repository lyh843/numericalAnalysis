from __future__ import annotations

import math
from typing import Callable

from config import SchedulerConfig, is_teacher


def constant_schedule(step: int, total_steps: int) -> float:
    return 1.0


def cosine_schedule(step: int, total_steps: int) -> float:
    return 0.5 * (1.0 + math.cos((step - 1) * math.pi / total_steps))


def warmup_cosine_schedule(step: int, total_steps: int) -> float:
    warmStep = 10
    if step <= warmStep:
        return 1.0 / warmStep * step
    else:
        return cosine_schedule(step - warmStep, total_steps - warmStep)


def step_decay_schedule(step: int, total_steps: int) -> float:
    # raise NotImplementedError("TODO: implement step_decay_schedule")
    changeStep = 5
    gamma = 0.8
    return 1.0 * gamma ** (math.floor((step - 1) / changeStep))


def build_scheduler(config: SchedulerConfig) -> Callable[[int, int], float]:
    name = config.name

    if name == "constant":
        return constant_schedule

    if is_teacher():
        from _teacher_solutions.schedulers import build_teacher_scheduler
        return build_teacher_scheduler(config)

    if name == "cosine":
        return cosine_schedule
    if name == "warmup_cosine":
        return warmup_cosine_schedule
    if name == "step_decay":
        return step_decay_schedule

    raise ValueError(f"Unknown scheduler name: {name}")
