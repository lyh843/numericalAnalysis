import numpy as np

x_0 = 1
x_1 = 3
x_2 = 2

def f(x):
    return x ** 3 - 3 * x - 1

def next_x(input_0, input_1, input_2):
    lambda_2 = (input_2 - input_1) / (input_1 - input_0)
    delta_2 = 1 + lambda_2
    a = f(input_0) * lambda_2 * lambda_2 - f(input_1) * lambda_2 * delta_2 + f(input_2) * lambda_2
    b = f(input_0) * lambda_2 * lambda_2 - f(input_1) * delta_2 * delta_2 + f(input_2) * (lambda_2 + delta_2)
    c = f(input_2) * delta_2
    den = b + np.sqrt(b * b - 4 * a * c) if np.abs(b + np.sqrt(b * b - 4 * a * c)) > np.abs(b - np.sqrt(b * b - 4 * a * c)) else b - np.sqrt(b * b - 4 * a * c)
    lambda_3 = -2 * c / den
    return input_2 + lambda_3 * (input_2 - input_1)

x_3 = next_x(x_0, x_1, x_2)
x_4 = next_x(x_1, x_2, x_3)
x_5 = next_x(x_2, x_3, x_4)

print(x_3)
print(x_4)
print(x_5)