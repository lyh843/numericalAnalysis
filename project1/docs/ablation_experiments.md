# 任务 1 说明（70 分）

任务 1 包含两部分：**代码实现**（任务 1.1，40 分）和**消融实验报告**（任务 1.2，30 分）。

---

## 任务 1.1：代码实现（40 分）

在 `student/` 目录下实现以下算法，每个文件中已提供接口定义和 `TODO` 标记，按提示补全即可。

| 模块 | 文件 | 具体内容 | 分值 |
| ---- | ---- | ---- | ---- |
| Loss 函数 | [student/losses.py](../student/losses.py) | 实现至少 3 种 loss（每种 2.5 分） | 7.5 |
| 初始化策略 | [student/initializers.py](../student/initializers.py) | 实现至少 2 种初始化策略（每种 2.5 分，见下方提示） | 5 |
| 优化器 | [student/optimizers.py](../student/optimizers.py) | 实现 SGD、SGD+Momentum、Adam、AdamW、Muon（每个 4 分），见下方参考 | 20 |
| 学习率调度器 | [student/schedulers.py](../student/schedulers.py) | 实现 Cosine Annealing、Warmup+Cosine、Step Decay（每种 2.5 分），见下方参考 | 7.5 |

#### 初始化策略提示

代码中预留了两种初始化策略的接口：`grid` 和 `image_sample`，你也可以自行设计其他策略。注意在本优化任务中，**初始化的质量对结果影响很大**。以下是两种参考思路：

- **`grid`（网格初始化）**：将 $N$ 个高斯均匀排布在图像上。这种策略确保高斯从一开始就均匀覆盖图像，避免 `random` 初始化中常见的"空洞"区域。
- **`image_sample`（图像采样初始化）**：利用目标图像的颜色信息来初始化高斯颜色，使初始渲染结果更接近目标。这种策略的核心优势是：高斯从第 0 步就拥有接近目标的颜色，优化器只需调整位置和尺度，大幅加速早期收敛。

#### 优化器参考

- **SGD**： $\theta_{t+1} = \theta_t - \eta \, g_t$ ；PyTorch 文档 [`SGD`](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)。**提示**：该优化器在此任务中表现不佳，请分析原因。
- **SGD+Momentum**： $v_{t+1} = \mu \, v_t + g_t$ ， $\theta_{t+1} = \theta_t - \eta \, v_{t+1}$ ；参考 Ruder (2016). [An overview of gradient descent optimization algorithms](https://arxiv.org/abs/1609.04747) Section 4.1
- **Adam**： $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ ， $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ ，用偏差修正 $\hat{m}_t, \hat{v}_t$ 后更新 $\theta_{t+1} = \theta_t - \eta \, \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$ ；PyTorch 文档 [`Adam`](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)；原文 Kingma & Ba (2015). [Adam](https://arxiv.org/abs/1412.6980) Algorithm 1
- **AdamW**：在 Adam 基础上将权重衰减从梯度中解耦，即先做 $\theta_t \leftarrow \theta_t (1 - \eta \lambda)$ ，再执行 Adam 更新；PyTorch 文档 [`AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)；原文 Loshchilov & Hutter (2019). [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) Algorithm 2
- **Muon**：Jordan, K. et al. (2024). [Muon: An optimizer for hidden layers in neural networks](https://github.com/KellerJordan/Muon)。核心是对梯度动量做 Newton-Schulz 正交化。注意：Muon 针对神经网络权重矩阵设计，本项目中参数形状为 `[N, 2]`/`[N, 3]`，列维度很小，正交化收益有限，效果预期介于 Momentum 和 Adam 之间。

#### 学习率调度器参考

- 可参考blog：https://zh.d2l.ai/chapter_optimization/lr-scheduler.html
- **Cosine Annealing**： $\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos\frac{t\pi}{T})$ ；PyTorch 文档 [`CosineAnnealingLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)；原文 Loshchilov & Hutter (2017). [SGDR](https://arxiv.org/abs/1608.03983) Eq. 5
- **Warmup+Cosine**：前 $T_w$ 步线性从 0 升至 $\eta_{max}$ ，之后按 Cosine Annealing 衰减；Goyal et al. (2017). [Accurate, Large Minibatch SGD](https://arxiv.org/abs/1706.02677) Section 2.2；PyTorch 文档 [`LinearLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html) + [`CosineAnnealingLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)
- **Step Decay**：每隔 $S$ 步将学习率乘以 $\gamma$ ，即 $\eta_t = \eta_0 \cdot \gamma^{\lfloor t/S \rfloor}$ ；PyTorch 文档 [`StepLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html)
- **提示**：本任务使用调度器的效果可能与在深度学习中的经验不同，请分析原因。

### 验收标准

- 代码能正常运行且结果合理即得满分，需在实验报告中报告结果。
- 核心公式与更新规则需要自行写出，不得直接调用 PyTorch 中对应的现成库函数（如 `torch.nn.functional` 中的 loss、`torch.optim` 中的优化器等）。
- `student_adam` 需与 `torch_adam` 结果接近（作为正确性校验）。

---

## 任务 1.2：消融实验报告（30 分）

### 实验原则

每次只改变一个模块，其余设置保持默认基线不变。如果需要改变默认基线，请在实验报告中说明并阐明原因。单个消融实验需保证除消融模块其他模块配置一致。

### 默认基线

| 项目 | 默认设置 |
| ---- | ---- |
| 图像 | `data/real_images/r1_flamingo_128.png` |
| 图像大小 | `128 x 128` |
| 高斯数量 | `1000` |
| 步数 | `200` |
| Loss | `mse` |
| Optimizer | `torch_adam` |
| Scheduler | `constant` |
| Initializer | `random` |
| 各向异性 | `True` |
| Alpha | `True` |
| 随机种子 | `42` |

### 五个消融实验（共 15 个实验，每个实验 2 分，共 30 分）

每个实验评分细则：

| 子项 | 分值 |
| ---- | ---- |
| 结果数据 + 可视化（表格、Loss 曲线、重建图等） | 1 |
| 分析讨论（设计是否有效的原因、收敛性、稳定性等） | 1 |

共性要求：

- 每次只改一个模块
- 其余设置保持默认基线
- 统一汇报 `PSNR / MSE / MAE`

| 实验 | 模块 | 需要做什么 | 比较对象 | 实验数 | 分值 | 其余保持不变 | 重点观察 | 补充说明 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 1.2A | Loss | 对比你在 1.1 中实现的 2 种 loss | `mse` vs 你选择的 2 种 loss | 2 | 4 | 初始化、优化器、调度器、模型开关 | 最终 PSNR、曲线平滑性、收敛速度 | 无 |
| 1.2B | 初始化策略 | 对比你在 1.1 中实现的初始化 | `random` vs 你的 2 种方法 | 2 | 4 | loss、优化器、调度器、模型开关 | 前 50 步收敛速度、最终 PSNR | 重点看前期收敛。此任务初始化的质量对结果影响很大。 |
| 1.2C | 优化器 | 对比你在 1.1 中实现的优化器 | `torch_adam`, `student_sgd`, `student_momentum`, `student_adam`, `student_adamw`, `student_muon` | 5 | 10 | 初始化、loss、调度器、模型开关 | 收敛速度、稳定性、最终 PSNR | `student_adam` 应与 `torch_adam` 接近，`sgd` 在此任务中表现不佳，请分析原因。 |
| 1.2D | 模型设计 | 无需实现新代码，仅切换配置 | 见下方 4 种组合 | 3 | 6 | 初始化、loss、优化器、调度器 | 各向异性和 alpha 对 PSNR 的贡献 | 比较开关是否互补 |
| 1.2E | 学习率调度器 | 对比你在 1.1 中实现的调度器 | `constant`, `cosine`, `warmup_cosine`, `step_decay` | 3 | 6 | 初始化、loss、优化器、模型开关 | 中后期收敛行为、最终 PSNR | 提示：本任务使用全量数据（非 minibatch）+ Adam 自适应学习率，调度器的效果可能与在深度学习中的经验不同，请分析原因 |

### 实验 1.2D 的 4 种组合

| 编号 | `use_anisotropic` | `use_alpha` | 说明 |
| ---- | ---- | ---- | ---- |
| D1 | `False` | `False` | 各向同性 + 无 alpha |
| D2 | `True` | `False` | 各向异性 + 无 alpha |
| D3 | `False` | `True` | 各向同性 + alpha |
| D4 | `True` | `True` | 各向异性 + alpha |

### 推荐的结果整理方式

| 图表 | 用途 |
| ---- | ---- |
| 总表格 | 汇总每组 `PSNR / MSE / MAE` |
| Loss 曲线图 | 比较收敛速度与稳定性 |
| 重建对比图 | 比较视觉质量 |
| 误差图 | 比较残差分布 |

代码提交要求详见 [README.md](../README.md) 中的提交要求部分。
