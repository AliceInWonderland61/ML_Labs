import numpy as np

a=np.array([0,1,2,3,4]) #1D array
b=np.array([[0,1,2], [3,4,5]]) #2D array
c=np.array([[[0,1,0],[2,3,2]],[[4,5,4],[6,7,6]]]) #3D array
print(np.shape(a))
print(np.size(a))
print(np.shape(b))
print(np.size(b))
print(np.shape(c))
print(np.size(c))

