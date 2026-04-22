import numpy as np
h = 439
H = 2384
R = 6371
a = (2 * R + H + h) / 2
c = (H - h) / 2


result = a * (np.pi / 12) * (1 + 4 * np.sqrt(1 - (c / a)**2 * 0.5) + np.sqrt(1 - (c / a)**2))
print(f"a = {a}\nc = {c}\nresult = {result}")