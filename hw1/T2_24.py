import numpy as np

x = np.array([0.25, 0.30, 0.39, 0.45, 0.53])
y = np.array([0.5000, 0.5477, 0.6245, 0.6708, 0.7280])

class T1:
    m_0 = 1.0000
    m_4 = 0.6868

class T2:
    m2_0 = 0
    m2_4 = 0

n = len(x) - 1

# ✅ 向量化计算
h = np.diff(x)
f = np.diff(y) / np.diff(x)

# lam, mu, g（只在 1~n-1 有效）
lam = np.zeros(n + 1)
mu = np.zeros(n + 1)
g = np.zeros(n + 1)

lam[1:n] = h[1:] / (h[:-1] + h[1:])
mu[1:n] = h[:-1] / (h[:-1] + h[1:])
g[1:n] = 3 * (lam[1:n] * f[:-1] + mu[1:n] * f[1:])

# =========================
# 题(1)：第一类边界条件
# =========================
t = T1

A1 = np.array([
    [2,       mu[1],    0],
    [lam[2],  2,        mu[2]],
    [0,       lam[3],   2]
])

C1 = np.array([
    g[1] - lam[1] * t.m_0,
    g[2],
    g[3] - mu[3] * t.m_4
])

B1 = np.linalg.solve(A1, C1)
print("题(1)的一阶导数值 [m1, m2, m3]:")
print(B1)

# =========================
# 题(2)：第二类边界条件
# =========================
A2 = np.zeros((n+1, n+1))

# 填三对角
np.fill_diagonal(A2, 2)
np.fill_diagonal(A2[1:], lam[1:])
np.fill_diagonal(A2[:,1:], mu[:-1])

# 边界条件
A2[0, 1] = 1
A2[-1, -2] = 1

C2 = g.copy()

B2 = np.linalg.solve(A2, C2)
print("\n题(2)的一阶导数值 [m0, m1, m2, m3, m4]:")
print(B2)