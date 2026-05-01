def f(x):
    return x ** 3 - 3 * x - 1

def f_1(x):
    return 3 * x ** 2 - 3

def next_x(x):
    return x - f(x) / f_1(x)

x_0 = 2
x_1 = next_x(x_0)
x_2 = next_x(x_1)
x_3 = next_x(x_2)
x_4 = next_x(x_3)
