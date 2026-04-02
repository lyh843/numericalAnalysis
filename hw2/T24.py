import matplotlib.pyplot as plt
t = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
y = [0, 1.27, 2.16, 2.86, 3.44, 3.87, 4.15, 4.37, 4.51, 4.58, 4.62, 4.64]

# plt.plot(t, y, marker='o')
# plt.title('Time vs. Distance')
# plt.xlabel('Time (s)')
# plt.ylabel('Distance (m)')
# plt.grid()
# plt.show()

# 感觉可以用 二次函数就能进行较好的拟合
import numpy as np
from scipy.optimize import curve_fit
def quadratic_model_two(t, a, b, c):
    return a * t**2 + b * t + c
params_two, _ = curve_fit(quadratic_model_two, t, y)
a, b, c = params_two
print(f"拟合的二次函数参数: a={a}, b={b}, c={c}")
def quadratic_model_three(t, a, b, c, d):
    return a * t**3 + b * t**2 + c * t + d
params_three, _ = curve_fit(quadratic_model_three, t, y)
a, b, c, d = params_three
print(f"拟合的三次函数参数: a={a}, b={b}, c={c}, d={d}")

# 绘制拟合曲线
t_fit = np.linspace(0, 55, 100)
y_fit_two = quadratic_model_two(t_fit, *params_two)
y_fit_three = quadratic_model_three(t_fit, *params_three)
plt.plot(t, y, 'o', label='Data')
plt.plot(t_fit, y_fit_two, label='at^2+bt+c', linestyle='--')
plt.plot(t_fit, y_fit_three, label='at^3+bt^2+ct+d', linestyle='-.')
plt.title('Time vs. Distance with Fits')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.legend()
plt.grid()
plt.show()