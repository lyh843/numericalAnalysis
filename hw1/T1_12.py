import numpy as np
sqrt_2 = 1.4

result0 = (np.sqrt(2) - 1) ** 6
result1 = 1 / ((sqrt_2 + 1) ** 6)
result2 = (3 - 2 * sqrt_2) ** 3
result3 = 1 / ((3 + 2 * sqrt_2) ** 3)
result4 = 99 - 70 * sqrt_2

e1 = np.abs(result1 - result0)
e2 = np.abs(result2 - result0)
e3 = np.abs(result3 - result0)
e4 = np.abs(result4 - result0)

print(e1)
print(e2)
print(e3)
print(e4)

if(e3 < e1 and e3 < e2 and e3 < e4):
    print("The best one is e3")