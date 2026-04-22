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

def printS(m, title):
    print(f"\n=== {title} ===")
    print("S(x) 的标准多项式形式（ax^3 + bx^2 + cx + d）：\n")
    
    for i in range(n):
        xi = x[i]
        hi = h[i]
        fi = f[i]
        
        # 原始系数（相对 (x - xi)）
        a = y[i]
        b = m[i]
        c = (3*fi - 2*m[i] - m[i+1]) / hi
        d = (m[i] + m[i+1] - 2*fi) / hi**2
        
        # 展开为 Ax^3 + Bx^2 + Cx + D
        A = d
        B = c - 3*d*xi
        C = b - 2*c*xi + 3*d*xi**2
        D = a - b*xi + c*xi**2 - d*xi**3
        
        xi1 = x[i+1]
        
        print(f"区间 [{xi:.2f}, {xi1:.2f}]：")
        print(f"S_{i}(x) = {A:.6f}x^3 + {B:.6f}x^2 + {C:.6f}x + {D:.6f}")
        print()

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

print("lam, mu, g:")
print(lam)
print(mu)
print(g)

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

A2 = np.array([
    [2, 1, 0, 0, 0],
    [lam[1], 2, mu[1], 0, 0],
    [0, lam[2], 2, mu[2], 0],    
    [0, 0, lam[3], 2, mu[3]],
    [0, 0, 0, 1, 2]
])


g[0] = 3 * f[0]
g[n] = 3 * f[n - 1]
C2 = g.copy()

B2 = np.linalg.solve(A2, C2)
print("\n题(2)的一阶导数值 [m0, m1, m2, m3, m4]:")
print(B2)

m1 = np.concatenate([[t.m_0], B1, [t.m_4]])
printS(m1, "题(1) - 第一类边界条件")

m2 = B2
printS(m2, "题(2) - 第二类边界条件")

# test1 = 0.25
# test2 = 0.3
# print(1.886295 * test1 ** 3 - 2.429036 * test1 ** 2 + 1.860838 * test1 + 0.157132)
# print(1.886295 * test2 ** 3 - 2.429036 * test2 ** 2 + 1.860838 * test2 + 0.157132)