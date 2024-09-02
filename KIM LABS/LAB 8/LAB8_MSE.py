import numpy as np

a_b=np.array([1.5,5.0])
data=np.array([[2.2,6.14],[1.3,4.72],[4.2,11.17],[5.8,14.23],[3.4,9.55],[8.7,22.49]])
x=data[:,0]
y=data[:,1]

mse=sum(((a_b[0]*x+a_b[1])-y)**2)/len(x)
print(mse)
