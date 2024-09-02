import numpy as np

data=np.array([[3.5,4.7,2.3,20.8],
               [4.4,5.7,4.1,29.1],
               [2.5,7.3,1.2,21.7],
               [8.5,3.3,4.8,30.5],
               [4.9,6.4,5.7,35.8],
               [7.2,7.1,7.4,44.6],
               [5.6,8.2,6.5,42.5]])

x=data[:, :-1]
y=data[:, -1]

w1,w2,w3=0,0,0
b=0

alpha=0.05

for i in range(10000):
    w1=w1-alpha*(1/len(data))*sum((x[:,0]*w1+x[:,1]*w2+x[:,2]*w3+b-y)*x[:,0])
    w2=w2-alpha*(1/len(data))*sum((x[:,0]*w1+x[:,1]*w2+x[:,2]*w3+b-y)*x[:,1])
    w3=w3-alpha*(1/len(data))*sum((x[:,0]*w1+x[:,1]*w2+x[:,2]*w3+b-y)*x[:,2])
    b=b-alpha*(1/len(data))*sum(x[:,0]*w1+x[:,1]*w2+x[:,2]*w3+b-y)

print("w1 =%f, w2= %f, w3= %f, b= %f" % (w1,w2,w3,b))
