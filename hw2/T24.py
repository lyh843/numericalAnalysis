import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

t = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
y = [0, 1.27, 2.16, 2.86, 3.44, 3.87, 4.15, 4.37, 4.51, 4.58, 4.62, 4.64]

def quadratic_model_two(t, a, b, c):
    return a * t**2 + b * t + c
params_two, _ = curve_fit(quadratic_model_two, t, y)
a, b, c = params_two
print(f"拟合的二次函数参数：a={a}, b={b}, c={c}")

def quadratic_model_three(t, a, b, c, d):
    return a * t**3 + b * t**2 + c * t + d
params_three, _ = curve_fit(quadratic_model_three, t, y)
a, b, c, d = params_three
print(f"拟合的三次函数参数：a={a}, b={b}, c={c}, d={d}")

def exp_model(t, a, b):
    return a * np.exp(b / t)
params_exp, _ = curve_fit(exp_model, t[1:], y[1:], p0=[1, 1])
a_exp, b_exp = params_exp
print(f"拟合的指数函数参数：a={a_exp}, b={b_exp}")

t_fit = np.linspace(0, 55, 100)
y_fit_two = quadratic_model_two(t_fit, *params_two)
y_fit_three = quadratic_model_three(t_fit, *params_three)
y_fit_exp = exp_model(t_fit, *params_exp)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(t, y, 'o', label='Data')
axes[0].plot(t_fit, y_fit_two, 'r--', label='at^2+bt+c')
axes[0].set_title('Quadratic Fit')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Distance (m)')
axes[0].legend()
axes[0].grid()

axes[1].plot(t, y, 'o', label='Data')
axes[1].plot(t_fit, y_fit_three, 'g-.', label='at^3+bt^2+ct+d')
axes[1].set_title('Cubic Fit')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Distance (m)')
axes[1].legend()
axes[1].grid()

axes[2].plot(t, y, 'o', label='Data')
axes[2].plot(t_fit, y_fit_exp, 'b:', label='a*exp(b/t)')
axes[2].set_title('Exponential Fit')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Distance (m)')
axes[2].legend()
axes[2].grid()

plt.tight_layout()
plt.show()
