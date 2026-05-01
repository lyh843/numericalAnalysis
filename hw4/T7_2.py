def f(x):
    return x ** 3 - 3 * x - 1

def next_x(x_k_1, x_k):
    return x_k - f(x_k) / (f(x_k) - f(x_k_1)) * (x_k - x_k_1)

x_0 = 2
x_1 = 1.9
x_2 = next_x(x_0, x_1)
x_3 = next_x(x_1, x_2)
x_4 = next_x(x_2, x_3)

print(f"x_2 = {x_2}\nx_3 = {x_3}\nx_4 = {x_4}")