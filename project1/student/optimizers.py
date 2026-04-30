from __future__ import annotations

import torch

from config import OptimizerConfig, is_teacher
from models import Gaussian2DModel


def build_torch_adam(param_groups: list[dict], lr: float) -> torch.optim.Optimizer:
    torch_groups = [{"params": g["params"], "lr": g["lr"], "base_lr": g["base_lr"]} for g in param_groups]
    return torch.optim.Adam(torch_groups, lr=lr)


class StudentSGD:
    def __init__(self, param_groups: list[dict]) -> None:
        self.param_groups = param_groups

    def zero_grad(self) -> None:
        raise NotImplementedError("TODO: implement SGD zero_grad()")

    def step(self) -> None:
        raise NotImplementedError("TODO: implement SGD step()")


class StudentMomentum:
    def __init__(self, param_groups: list[dict]) -> None:
        self.param_groups = param_groups
        self.velocity: dict[int, torch.Tensor] = {}

    def zero_grad(self) -> None:
        raise NotImplementedError("TODO: implement Momentum zero_grad()")

    def step(self) -> None:
        raise NotImplementedError("TODO: implement Momentum step()")


class StudentAdam:
    def __init__(self, param_groups: list[dict]) -> None:
        self.param_groups = param_groups
        self.step_count = 0
        self.state: dict[int, dict[str, torch.Tensor]] = {}

    def zero_grad(self) -> None:
        raise NotImplementedError("TODO: implement Adam zero_grad()")

    def step(self) -> None:
        raise NotImplementedError("TODO: implement Adam step()")


class StudentAdamW:
    def __init__(self, param_groups: list[dict]) -> None:
        self.param_groups = param_groups
        self.step_count = 0
        self.state: dict[int, dict[str, torch.Tensor]] = {}

    def zero_grad(self) -> None:
        raise NotImplementedError("TODO: implement AdamW zero_grad()")

    def step(self) -> None:
        raise NotImplementedError("TODO: implement AdamW step()")


class StudentMuon:
    def __init__(self, param_groups: list[dict]) -> None:
        self.param_groups = param_groups
        self.buffers: dict[int, torch.Tensor] = {}

    def zero_grad(self) -> None:
        raise NotImplementedError("TODO: implement Muon zero_grad()")

    def step(self) -> None:
        raise NotImplementedError("TODO: implement Muon step()")


def build_optimizer(model: Gaussian2DModel, config: OptimizerConfig):
    name = config.name
    base_lr = config.lr
    param_groups = model.get_param_groups(base_lr, config.param_groups)

    if name == "torch_adam":
        return build_torch_adam(param_groups=param_groups, lr=base_lr)

    if name == "student_sgd":
        if is_teacher():
            from _teacher_solutions.student_optimizers import StudentSGD as TeacherStudentSGD
            return TeacherStudentSGD(param_groups=param_groups)
        return StudentSGD(param_groups=param_groups)

    if name == "student_momentum":
        if is_teacher():
            from _teacher_solutions.student_optimizers import StudentMomentum as TeacherStudentMomentum
            return TeacherStudentMomentum(param_groups=param_groups)
        return StudentMomentum(param_groups=param_groups)

    if name == "student_adam":
        if is_teacher():
            from _teacher_solutions.student_optimizers import StudentAdam as TeacherStudentAdam
            return TeacherStudentAdam(param_groups=param_groups)
        return StudentAdam(param_groups=param_groups)

    if name == "student_adamw":
        if is_teacher():
            from _teacher_solutions.student_optimizers import StudentAdamW as TeacherStudentAdamW
            return TeacherStudentAdamW(param_groups=param_groups)
        return StudentAdamW(param_groups=param_groups)

    if name == "student_muon":
        if is_teacher():
            from _teacher_solutions.student_optimizers import StudentMuon as TeacherStudentMuon
            return TeacherStudentMuon(param_groups=param_groups)
        return StudentMuon(param_groups=param_groups)

    raise ValueError(f"Unknown optimizer name: {name}")
