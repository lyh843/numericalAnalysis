import numpy as np
x = [-0.9062, -0.5385, 0, 0.5385, 0.9062]
A = [0.2369, 0.4786, 0.5689, 0.4786, 0.2369]

def f(x: float):
    return 1 / (x + 2)

result = 0.0

for i in range(5):
    result += A[i] * f(x[i])
    
print(result)


x_pro = [-0.9061798459, -0.5384693101, 0, 0.5384693101, 0.9061798459]
A_pro = [0.2369268851, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268851]

result_pro = 0.0

for i in range(5):
    result_pro += A_pro[i] * f(x_pro[i])
    
print(result_pro)

a = 1 / np.sqrt(3)

def g(x: float):
    return 1 / x

result_new = g(5 + a) + g(5 - a) + g(7 + a) + g(7 - a) + g(9 + a) + g(9 - a) + g(11 + a) + g(11 - a)

print(result_new)