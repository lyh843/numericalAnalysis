### 损失函数

#### 1. MSE Loss

**公式：**
$$
L = \frac{1}{N} \sum_{i = 1}^n \left( y_i - \hat{y_i} \right)^2
$$

#### 2. L1 Loss

**公式：** 
$$
L = \frac{1}{N} \sum_{i = 1}^n \left| y_i - \hat{y_i} \right|
$$

#### 3. Charbonnier Loss

**公式：**
$$
L = \frac{1}{N} \sum_{i = 1}^n \sqrt{(y_i - \hat{y_i})^2 + \epsilon}
$$

#### 4. MSE L1 Loss

**公式：**
$$
L = \alpha \frac{1}{N} \sum_{i = 1}^n(y_i - \hat{y_i})^2 + \beta \frac{1}{N} \sum_{i = 1}^n \left|y_i - \hat{y_i} \right|
$$

#### 5. MSE Edge Loss

**公式：**
$$
L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 \, + \, \lambda \cdot \frac{1}{N} \sum_{i=1}^{N} \left\| \nabla y_i - \nabla \hat{y}_i \right\|^2
$$

---

### 优化器

#### 1. SGD:

**公式：**
$$
\theta_{t+1} = \theta_t - \eta g_t
$$
其中$\eta$是学习率，$g_t$是当前参数的梯度，也就是`loss`增长最快的方向。

#### 2. SGD + Momentum

**公式：**
$$
v_{t+1} = \mu v_t + g_t, \quad \theta_{t + 1} = \theta_t - \eta v_{t + 1}
$$
引入了动量、速度的概念，其中$\mu$一般设为$0.9$

#### 3. Adam

**公式：**

一阶动量：
$$
m_{t + 1} = \beta_1 m_t + (1 - \beta_1)g_t
$$
二阶动量：
$$
v_{t + 1} = \beta_2 v_t + (1 - \beta_2)g_t^2
$$
偏差修正：
$$
\hat{m_t} = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v_t} = \frac{v_t}{1 - \beta_2^t}
$$
更新公式：
$$
\theta_{t + 1} = \theta_t - \eta \frac{\hat{m_t}}{\sqrt{\hat{v_t}}+ \epsilon}
$$

#### 4. AdamW

更新公式：
$$
\theta_{t + 1} = \theta_t - \eta \frac{\hat{m_t}}{\sqrt{\hat{v_t}}+ \epsilon} - \eta \lambda \theta_t
$$

#### 5. Muon



---

### 初始化

### Grid 网格初始化

根据得到的`num_gaussians`去开方得到近似的行数和列数，然后生成相应个数的`x`和`y`点，`center_init`最后就是`x`和`y`的两个数列中元素逐个组合的结果。

其余参数沿用随机初始化的代码。



---

### 实验结果

初始模块配置：

| 项目     | 图像大小       | 高斯数量 | 步数  | Loss  | Optimizer    | Scheduler  | Initializer | 各向异性 | Alpha  | 随机种子 |
| -------- | -------------- | -------- | ----- | ----- | ------------ | ---------- | ----------- | -------- | ------ | -------- |
| 默认配置 | $128\times128$ | $1000$   | $200$ | `mse` | `torch_adam` | `constant` | `random`    | `True`   | `True` | $42$     |

#### LOSS 模块消融实验





## 参考资料



