import numpy as np

arr0 = np.zeros((2,2))
arr1 = np.ones((2,2))

rows = [ np.hstack([arr0,arr1]), np.hstack((arr1,arr0))]

print(np.vstack(rows))