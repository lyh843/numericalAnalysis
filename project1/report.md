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
