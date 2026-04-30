# 实验 1：CM2026 Project 1

本实验的任务是给定一张目标图像，通过优化调整一组 2D Gaussian点云的参数来对图像进行拟合，在此过程中你将实现优化系统的各个模块（损失函数、初始化方法、优化器、学习率策略），并比较不同设计对优化效果的影响。

本仓库提供了可运行基线，以及 loss、初始化策略、优化器、学习率调度器等扩展接口，供任务 1 和任务 2 使用。优化问题的形式化定义见 [docs/problem_formulation.md](docs/problem_formulation.md)。

下图展示了 `flamingo` 目标图像在 `image_sample` 初始化下的优化过程：

<p align="center">
  <img src="docs/teaser.gif" width="720" alt="Optimization teaser">
  <br>
  <em>flamingo，image_sample 初始化，优化过程动画</em>
</p>

## 一、快速开始

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 运行默认训练：

```bash
python train.py
```

默认配置定义在 [config.py](config.py) 中。

运行成功训练结束后，结果会默认保存到 `outputs/`。通常会包含目标图、最终重建图、若干中间结果、训练曲线以及最终评估结果。如果开启 `config.train.save_video`，还会额外导出优化过程动画和对应的视频帧。下图展示了默认配置（r1_flamingo，1000 个高斯，random 初始化，MSE loss，仓库内置的 `torch_adam` 参考基线，其内部调用 `torch.optim.Adam`）在 200 步训练后的效果：

<p align="center">
  <img src="docs/comparison_baseline.png" width="720" alt="Baseline comparison">
  <br>
  <em>左：r1_flamingo 目标图像 | 中：random 基线重建 | 右：绝对误差</em>
</p>


## 二、实验要求

### 任务概览

本实验包含两个任务：

- **任务 1**（代码实现 + 消融实验，70 分）：在 `student/` 目录下实现算法，并逐模块对比不同算法对重建效果的影响。详见 [docs/ablation_experiments.md](docs/ablation_experiments.md)。
- **任务 2**（配置优化，30 分）：自由组合上述模块与超参数，最大化 PSNR。详见 [docs/assignment2.md](docs/assignment2.md)。

### 实现约束

- 核心公式与更新规则需要自行写出，不得直接调用 PyTorch 中对应的现成库函数（如 `torch.nn.functional` 中的 loss、`torch.optim` 中的优化器等）。
- 可以继续使用 PyTorch 的张量运算、自动求导和基础数学操作。

### 建议完成顺序

1. 运行默认基线，确认训练流程与输出结果正常。
2. 补全 `student/` 下的四个文件。
3. 按模块完成任务 1 的消融实验。
4. 整理最佳配置并完成任务 2。

### 评分构成

| 部分 | 分值 |
| ---- | ---- |
| **任务 1：代码实现 + 消融实验** | **70** |
| &emsp;任务 1.1：代码实现 | 40 |
| &emsp;&emsp;损失函数实现（3 × 2.5） | 7.5 |
| &emsp;&emsp;优化器实现（5 × 4） | 20 |
| &emsp;&emsp;初始化器实现（2 × 2.5） | 5 |
| &emsp;&emsp;学习率调度器实现（3 × 2.5） | 7.5 |
| &emsp;任务 1.2：消融实验报告（15 × 2） | 30 |
| &emsp;&emsp;1.2A：Loss 函数消融 | 4 |
| &emsp;&emsp;1.2B：初始化策略消融 | 4 |
| &emsp;&emsp;1.2C：优化器消融 | 10 |
| &emsp;&emsp;1.2D：模型设计消融 | 6 |
| &emsp;&emsp;1.2E：学习率调度器消融 | 6 |
| **任务 2：配置优化** | **30** |
| &emsp;任务 2A：100 步迭代优化 | 15 |
| &emsp;任务 2B：500 步迭代优化 | 15 |
| **合计** | **100** |

## 三、提交要求

### 1. 实验报告

实验报告请以 `pdf` 格式提交。建议至少包含以下内容：

- 实验设置：简要说明你使用的基线配置，以及实现了哪些模块。
- 任务 1.2 结果：按 1.2A–1.2E 五个部分整理消融实验结果。每部分至少给出结果表，并统一汇报 `PSNR / MSE / MAE`。
- 结果分析：结合 loss 曲线、重建图、误差图分析不同方法的收敛速度、稳定性和最终效果。并分析结果的可靠性，以及产生这样结果的原因。
- 任务 2 结果：单独汇报任务 2A（100 步）和任务 2B（500 步）的结果，并简要说明最终采用的设计及原因。
- 总结：概括哪些设计有效，哪些设计效果一般，以及你对该任务的主要观察。

### 2. 代码

代码部分应保证可直接运行。提交文件说明如下：

| 文件 | 是否必交 | 说明 |
| ---- | ---- | ---- |
| `student/losses.py` | **必交** | 任务 1.1：loss 实现 |
| `student/optimizers.py` | **必交** | 任务 1.1：优化器实现 |
| `student/initializers.py` | **必交** | 任务 1.1：初始化策略实现 |
| `student/schedulers.py` | **必交** | 任务 1.1：调度器实现 |
| `experiments/assignment2_settings.py` | **必交** | 任务 2 配置 |
| 其他新增的 `student/*.py` | 按需提交 | 如果你新增了辅助模块，需一并提交 |

即使某部分实验没有完全做完，上述必交文件仍需全部提交（可保留未实现的 stub），以便统一验收。不需要提交 `outputs/` 下的结果文件，图表和数值直接整理进报告即可。

### 3. 压缩包组织示例

```text
姓名_学号_Project1_v1.zip
|
|-- report.pdf
|
|-- code/
|   |-- experiments/
|   |   |-- assignment2_settings.py
|   |-- student/
|   |   |-- losses.py
|   |   |-- optimizers.py
|   |   |-- initializers.py
|   |   |-- schedulers.py
```

## 四、文件说明

| 文件 | 说明 |
| ---- | ---- |
| [config.py](config.py) | 全局配置：目标图、模型、训练、loss、优化器、调度器等参数 |
| [train.py](train.py) | 训练主循环 |
| [models.py](models.py) | 2D Gaussian 模型定义与参数约束 |
| [renderer.py](renderer.py) | 可微分高斯渲染器 |
| [target_generators.py](target_generators.py) | 目标图像加载与合成 |
| [student/losses.py](student/losses.py) | **作业**：实现至少 3 种 loss |
| [student/optimizers.py](student/optimizers.py) | **作业**：实现 SGD、Momentum、Adam、AdamW、Muon |
| [student/initializers.py](student/initializers.py) | **作业**：实现高斯初始化方法 |
| [student/schedulers.py](student/schedulers.py) | **作业**：实现 `cosine`、`warmup_cosine`、`step_decay` |
| [experiments/assignment2_settings.py](experiments/assignment2_settings.py) | 任务 2 配置（直接编辑此文件） |
| [experiments/run_assignment2.py](experiments/run_assignment2.py) | 任务 2 本地自测脚本 |


## 五、任务 2 自测

1. 编辑 [experiments/assignment2_settings.py](experiments/assignment2_settings.py)，调整 `get_sprint_setting()`（任务 2A，100 步）和 `get_standard_setting()`（任务 2B，500 步）中的配置。硬约束字段已锁定，可调字段包括 loss、initializer、optimizer、scheduler、lr、param_groups 倍率、模型开关等，详见文件中的注释。

2. 运行自测：

```bash
python experiments/run_assignment2.py --track both
```

- `--track sprint` / `--track standard`：只跑单个任务。
- `--limit 2`：只跑前 2 张图，适合快速验证。

3. 终端输出每张图的 PSNR 和平均值，结果保存在 `outputs_assignment2/`。对照 [docs/assignment2.md](docs/assignment2.md) 中的阈值表估算得分。


## 补充说明

- 任务 2 采用固定 PSNR 阈值评分，具体规则见 [docs/assignment2.md](docs/assignment2.md)。
- 如需更细致的实验对比，可自行扩展可视化，仅供报告中的图像制作、帮助自己调试，无需上传这部分代码。
- 任务 2 的合成数据集源文件已经提供，方便同学调试，但不得利用数据作弊，包括但不限于：直接读取数据集的txt文件初始化，在代码中写大量的针对数据的硬编码。
- 助教将会抽查提交的代码。对于与报告内容严重不符、作弊、违反诚信的内容予以严肃处理。
- 本库中文档与代码发生矛盾时，以文档内容优先。

## 参考文献

- **3D Gaussian Splatting**: Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. ACM Transactions on Graphics, 42(4). https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **Adam**: Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR 2015. https://arxiv.org/abs/1412.6980
- **Muon**: Jordan, K. et al. (2024). *Muon: An optimizer for hidden layers in neural networks*. https://github.com/KellerJordan/Muon

## 学术诚信

请独立完成本次作业。

- 可以阅读课程提供的代码、文档与参考资料。
- 可以与同学讨论思路，但不得直接交换代码、实验结果或报告文本。
- 不得抄袭或改写他人实现后冒充为自己的工作。
- 如使用课外资料、工具或生成式 AI，请遵守课程要求并如实说明用途。
- 提交的代码、结果和分析必须与本人实际实现一致。如出现报告数值与本人实现不一致的情况，将严肃处理。

祝实验顺利！
