import pandas as panda
import numpy as np
from sklearn.model_selection import train_test_split


R = panda.read_csv("./iris/iris.data", sep=',', header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
R = R.dropna()

map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
R['class'] = R['class'].map(map)

counter = 0

for i in range(100):

    shuffle = R.sample(frac=1).reset_index(drop=True)

    # splitting the datasets into the training part and test
    train, test = train_test_split(shuffle, test_size=0.2, random_state=i)

    
    train_x = train.iloc[:, :4].values
    train_y = panda.get_dummies(train.iloc[:, 4]).values

    # features for testing (4)
    test_x = test.iloc[:, :4].values
    test_y = panda.get_dummies(test.iloc[:, 4]).values

    # Weights
    W = np.zeros((4, 3))
    B = np.zeros(3)
    alpha = 0.005
    N = train_x.shape[0]
    c = W.shape[1]


    for i in range(1000):
        for j in range(c):
            W[:, j] = W[:, j] - alpha * (1 / N) * np.dot(np.transpose(np.dot(train_x, W[:, j]) + B[j] - train_y[:, j]),
                                                         train_x)
            B[j] = B[j] - alpha * (1 / N) * sum(np.dot(train_x, W[:, j]) + B[j] - train_y[:, j])

    counter += np.sum(np.argmax(np.dot(test_x, W) + B, axis=1) == np.argmax(test_y, axis=1)) / test_x.shape[0]
    print(counter)
# average over 100 loops
accuracy = counter / 100

print("Accuracy: ", accuracy)
