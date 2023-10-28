import numpy as np

# create a 2 * 3 * 3 + 2 array
a = np.arange(2 * 3 * 3 + 2).reshape(2, 3, 3) + 1
print(a)