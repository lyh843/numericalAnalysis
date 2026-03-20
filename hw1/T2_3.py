import numpy as np
# 线性插值
f = {
        0.4: -0.916291, 
        0.5: -0.693147,
        0.6: -0.510826,
        0.7: -0.357765,
        0.8: -0.223144
        }

k = (f.get(0.6) - f.get(0.5)) / (0.6 - 0.5)
b = f.get(0.5) - k * 0.5

result1 = k * 0.54 + b
print(f"线性插值结果：{result1}")
    
print(6.93147 - 5.10826)
print(5.10826 * 0.5 - 6.93147 * 0.6)
print(0.54 * 1.82321 - 1.604752)

# 二次插值

x = np.array([0.5, 0.6, 0.7])
y = np.array([-0.693147, -0.510826, -0.357765])

A = np.vstack([x**2, x, np.ones(len(x))]).T

a, b, c = np.linalg.solve(A, y)

result2 = a * (0.54 ** 2) + b * 0.54 + c
print(f"二次插值结果：{result2}")

a_acc = 50 * f.get(0.5) - 100 * f.get(0.6) + 50 * f.get(0.7)
b_acc = -1.3 * 50 * f.get(0.5) + 1.2 * 100 * f.get(0.6) - 1.1 * 50 * f.get(0.7)
c_acc = 0.42 * 50 * f.get(0.5) - 0.35 * 100 * f.get(0.6) + 0.3 * 50 * f.get(0.7)

result_acc = a_acc * (0.54 ** 2) + b_acc * 0.54 + c
print(f"a = {a_acc}")
print(f"b = {b_acc}")
print(f"c = {c_acc}")
print(result_acc)
print(np.log(0.54))