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
        # raise NotImplementedError("TODO: implement SGD zero_grad()")
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.zero_()
                

    def step(self) -> None:
        # raise NotImplementedError("TODO: implement SGD step()")
        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    param -= lr * param.grad


class StudentMomentum:
    def __init__(self, param_groups: list[dict]) -> None:
        self.param_groups = param_groups
        self.velocity: dict[int, torch.Tensor] = {}

    def zero_grad(self) -> None:
        # raise NotImplementedError("TODO: implement Momentum zero_grad()")
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.zero_()

    def step(self) -> None:
        # raise NotImplementedError("TODO: implement Momentum step()")
        mu = 0.9
        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    pid = id(param)
                    if pid not in self.velocity:
                        self.velocity[pid] = torch.zeros_like(param)
                    
                    v = mu * self.velocity[pid] + param.grad
                    self.velocity[pid] = v
                    param -= lr * v
                    
                    
class StudentAdam:
    def __init__(self, param_groups: list[dict]) -> None:
        self.param_groups = param_groups
        self.step_count = 0
        self.state: dict[int, dict[str, torch.Tensor]] = {}

    def zero_grad(self) -> None:
        # raise NotImplementedError("TODO: implement Adam zero_grad()")
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.zero_()

    def step(self) -> None:
        # raise NotImplementedError("TODO: implement Adam step()")
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        
        self.step_count += 1

        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    
                    pid = id(param)
                    if pid not in self.state:
                        self.state[pid] = {
                            "m": torch.zeros_like(param),
                            "v": torch.zeros_like(param),
                        }
                    
                    grad = param.grad
                    m = beta1 * self.state[pid]["m"] + (1 - beta1) * grad
                    v = beta2 * self.state[pid]["v"] + (1 - beta2) * grad * grad
                    self.state[pid]["m"] = m
                    self.state[pid]["v"] = v
                    
                    m_hat = m / (1 - beta1 ** self.step_count)
                    v_hat = v / (1 - beta2 ** self.step_count)
                    
                    param -= lr * m_hat / (torch.sqrt(v_hat) + epsilon)
                    
                    

class StudentAdamW:
    def __init__(self, param_groups: list[dict]) -> None:
        self.param_groups = param_groups
        self.step_count = 0
        self.state: dict[int, dict[str, torch.Tensor]] = {}

    def zero_grad(self) -> None:
        # raise NotImplementedError("TODO: implement AdamW zero_grad()")
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.zero_()


    def step(self) -> None:
        # raise NotImplementedError("TODO: implement AdamW step()")
        lam = 0.01
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        
        self.step_count += 1

        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    
                    pid = id(param)
                    if pid not in self.state:
                        self.state[pid] = {
                            "m": torch.zeros_like(param),
                            "v": torch.zeros_like(param),
                        }
                    
                    grad = param.grad
                    m = beta1 * self.state[pid]["m"] + (1 - beta1) * grad
                    v = beta2 * self.state[pid]["v"] + (1 - beta2) * grad * grad
                    self.state[pid]["m"] = m
                    self.state[pid]["v"] = v
                    
                    m_hat = m / (1 - beta1 ** self.step_count)
                    v_hat = v / (1 - beta2 ** self.step_count)
                    
                    param -= (lr * m_hat / (torch.sqrt(v_hat) + epsilon) + lr * lam * param)
                    

class StudentMuon:
    def __init__(self, param_groups: list[dict]) -> None:
        self.param_groups = param_groups
        self.buffers: dict[int, torch.Tensor] = {}

    @staticmethod
    def _orthogonalize(update: torch.Tensor, num_steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
        if update.ndim == 1:
            return update / (update.norm() + eps)

        original_shape = update.shape
        x = update.reshape(update.shape[0], -1)
        transposed = False
        if x.shape[0] < x.shape[1]:
            x = x.transpose(0, 1)
            transposed = True

        x = x / (x.norm() + eps)
        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(num_steps):
            gram = x @ x.transpose(0, 1)
            x = a * x + (b * gram + c * (gram @ gram)) @ x

        if transposed:
            x = x.transpose(0, 1)
        return x.reshape(original_shape)

    def zero_grad(self) -> None:
        # raise NotImplementedError("TODO: implement Muon zero_grad()")
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.zero_()

    def step(self) -> None:
        momentum = 0.95

        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                for param in group["params"]:
                    if param.grad is None:
                        continue

                    pid = id(param)
                    if pid not in self.buffers:
                        self.buffers[pid] = torch.zeros_like(param)

                    buf = self.buffers[pid]
                    buf.mul_(momentum).add_(param.grad)
                    update = self._orthogonalize(buf)
                    param -= lr * update


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
