import pandas as panda
import numpy as np
import random

R=panda.read_csv('./iris/iris.data', sep=',', header=None)
#print(R)

R.head()
R.replace('Iris-setosa', np.nan, inplace=True)
R.dropna(inplace=True)
R.replace('Iris-versicolor', np.nan, inplace=True)
R.dropna(inplace=True)

R.iloc[:, 4]=panda.factorize(R.iloc[:, 4])[0]
LC_Data=R.values


def hypothesis(X,w,b):
    return np.dot(X,w)+b

counter=0
for i in range(100):
#trainx=np.array(R.iloc[0])
#Gotta shuffle the data
    random.shuffle(LC_Data)
    #subset = LC_Data[:100, :]
    #subset = subset[subset[:, -1] < 2, :]

    #train_x=R[:0]
    #train_y=R[:1]
# We want 100 samples and we have 4 attributes
    train_x = LC_Data[:100, :4]
    train_y = LC_Data[:100, 4]

    test_x = LC_Data[:100, :4]
    test_y = LC_Data[:100, 4]

    w=np.zeros(np.size(train_x,1))
    b=0
    alpha =0.05

    for k in range(100):
        w=w-alpha*(1/len(train_x))*np.dot(np.transpose(np.dot(train_x,w)+b-train_y),train_x)
        b=b-alpha*(1/len(train_x))*np.sum(np.dot(train_x,w)+b-train_y)
   # print("w =%f, b=%f" % (w, b))
    #test_x = subset[:, :2].values
    #test_y = subset[:, -1].values

    counter+=sum(np.sign(hypothesis(test_x,w,b)) == test_y) / len(test_x)
    accuracy=counter/100
print("Accuracy:", accuracy)