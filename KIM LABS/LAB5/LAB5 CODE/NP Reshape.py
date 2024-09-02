import numpy as np
a=np.array([0,1,2,3,4,5,6,7])
b=a.reshape(2,4)
print(b)
c=np.reshape(a,(4,-1))
print(c)