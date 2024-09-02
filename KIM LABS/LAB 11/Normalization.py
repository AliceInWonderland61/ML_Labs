from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ucimlrepo import fetch_ucirepo
import pandas as panda
import numpy as np
# r is raw string and instead of using tabs just use the string + to avoid errors
R = panda.read_csv('./auto+mpg/auto-mpg.data', sep=r'\s+', header=None)

R.head()
R.replace('?', np.nan, inplace=True)
R.dropna(inplace=True)

print(R)
# X=[[1,2],[2,1],[3,2],[4,3]]
scaler = MinMaxScaler()
n_data = scaler.fit_transform(R.iloc[:, 1:8])

print(n_data)

R=np.concatenate((R.iloc[:,:1], n_data), axis=1)

print(R)


