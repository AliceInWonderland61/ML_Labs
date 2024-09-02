import numpy as np

a=np.array([0,1,2,3])
b=np.array([2,3,1,3])
print(np.dot(a,b))
c=np.array([0,1,2,3,4,5]).reshape(2,3)
d=np.array([0,1,2,3,4,5]).reshape(3,2)
print(np.dot(c,d))