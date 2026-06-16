import numpy as np

D = np.array([[5, 0, 0], [0, 4, 0], [0, 0, 10]])

L = -np.array([[0, 0, 0], [-1, 0, 0], [2, -3, 0]])

U = -np.array([[0, 2, 1], [0, 0, 2], [0, 0, 0]])

b = np.array([-12, 20, 3])

w = 0.9

def format_array(arr):
    return np.array2string(arr, formatter={"float_kind": lambda x: f"{x:.6f}"})

def inf_norm(vec):
    max_abs = 0.0
    for value in vec:
        abs_value = abs(value)
        if abs_value > max_abs:
            max_abs = abs_value
    return max_abs

Lw = np.linalg.inv(D - w * L) @ ((1 - w) * D + w * U)

print("Lw =")
print(format_array(Lw))

f = w * np.linalg.inv(D - w * L) @ b

print("f =")
print(format_array(f))

x_prev = np.array([0.0, 0.0, 0.0])

for k in range(1, 10):
    x_curr = Lw @ x_prev + f
    diff = x_curr - x_prev
    x_norm_inf = inf_norm(diff)

    print(f"x_{k} =")
    print(format_array(x_curr))
    print(f"||x_{k} - x_{k - 1}||_inf = {x_norm_inf:.6f}")

    x_prev = x_curr
