import pandas as panda
import numpy as np

R = panda.read_csv("./iris/iris.data", sep=',', header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
subclass = R.dropna()
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Assuming binary classification (combining classes)
subclass['class'] = subclass['class'].apply(lambda x: 1 if x == 'Iris-setosa' else 0)

counter = 0

for i in range(100):
    subclass = subclass.sample(frac=1).reset_index(drop=True)
    total_rows = subclass.shape[0]
    train_size = int(total_rows * 0.8)

    train = subclass.loc[:train_size - 1, :]
    test = subclass.loc[train_size:, :]

    train_x = train.iloc[:, :4].values
    train_y = train.iloc[:, 4].values

    test_x = test.iloc[:, :4].values
    test_y = test.iloc[:, 4].values

    w = np.random.rand(train_x.shape[1])
    lr = 0.05

    for k in range(100):

        w_diff = np.dot(np.transpose(train_y - sigmoid(np.dot(train_x, w))), train_x)
        w = w + lr * w_diff

    counter += sum(np.round(sigmoid(np.dot(test_x, w))) == test_y) / np.size(test_y)
    print(counter)

accuracy = counter / 100

print("Accuracy: ", accuracy)
