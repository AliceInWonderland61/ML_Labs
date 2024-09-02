import numpy as np

a=np.array([[0,1],[2,3]])
print(np.sum(a))
print(np.sum(a, axis=0))
print(np.sum(a, axis=1))
print(np.sum(a, axis=1, keepdims=True))
print(np.max(a))
print(np.argmax(a, axis=0))