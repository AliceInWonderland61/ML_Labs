import pandas as panda
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.adam import Adam
from sklearn.model_selection import train_test_split

#Lab 15 XOR code
R = panda.read_csv("./titanic/train.csv")
model = nn.Sequential(
    nn.Linear(15, 4),
    nn.Sigmoid(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)
#We don't really need these so we can drop them
R.head()
R.replace('PassengerId', np.nan, inplace=True)
R.dropna(inplace=True)
R.replace('Name', np.nan, inplace=True)
R.dropna(inplace=True)
R.replace('Ticket', np.nan, inplace=True)
R.dropna(inplace=True)
R.replace('Fare', np.nan, inplace=True)
R.dropna(inplace=True)
R.replace('Embarked', np.nan, inplace=True)
R.dropna(inplace=True)

#save the rest to titanic_a
titanic_a = R
#gather the male and female columns
titanic_a['Male'] = titanic_a['Sex'].apply(lambda x: 1 if x == 'male' else 0)
titanic_a['Female'] = titanic_a['Sex'].apply(lambda x: 1 if x == 'female' else 0)
titanic_a.drop('Sex', axis=1, inplace=True)

#there's missing values in the csv for age, so we fill them up with 0s
titanic_a['Age'] = titanic_a['Age'].fillna(0)

#C_Exists checking if the info is there
titanic_a['C_Exists'] = titanic_a['Cabin'].apply(lambda x: 0 if panda.isna(x) else 1)
titanic_a['Cabin'] = titanic_a['Cabin'].fillna(0)


rows_data = []
for _, row in titanic_a.iterrows():
    if row['Cabin'] != 0:
        cabins = str(row['Cabin']).split(' ')
        for cabin in cabins:
            new_row = row.copy()
            new_row['Cabin'] = cabin
            rows_data.append(new_row)
    else:
        rows_data.append(row)

titanic_a = panda.DataFrame(rows_data)
#CABIN COUNT
count_c = titanic_a['Cabin'].nunique()

panda.set_option('display.max_rows', 100)
cabin_data = titanic_a['Cabin']
cabin_data = cabin_data.head(20)
#cabin entries
entries_c = titanic_a['Cabin'].unique()


titanic_a['Cabin'] = titanic_a['Cabin'].astype(str)

#cabin contains lettes A-G and T so create an array with those columns
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']


for i in letters:
    titanic_a[f'{i}_cabin'] = titanic_a['Cabin'].apply(lambda x: 1 if x.startswith(i) else 0)


#don't need Cabin anymore so it'll be dropped; the original one we had
titanic_a.drop('Cabin', axis=1, inplace=True)

#don't need this rn
TX = titanic_a.drop('Survived', axis=1)
Ty = titanic_a['Survived']


TX = titanic_a[['Pclass', 'Age', 'SibSp', 'Parch', 'Male', 'Female',
                'C_Exists', 'A_cabin', 'B_cabin', 'C_cabin', 'D_cabin', 'E_cabin',
                'F_cabin', 'G_cabin', 'T_cabin']]
Ty = titanic_a['Survived']

#splitting the data for the train and test parts
X_train, X_test, y_train, y_test = train_test_split(TX, Ty, test_size=0.2, random_state=42)

#Lab 15 XOR code
x_train_tensor = torch.FloatTensor(X_train.values)
y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
#Lab 15 XOR code
optim = Adam(model.parameters(), lr=0.01)

#Lab 15 XOR code
for epoch in range(1000):
    optim.zero_grad()
    preds = model(x_train_tensor)
    loss = nn.BCELoss()(preds, y_train_tensor)
    loss.backward()
    optim.step()
    if epoch % 100 == 0:
        print('loss:', loss.item())
        print(' ')


#Lab 15 XOR code
X_test_tensor = torch.FloatTensor(X_test.values)
y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1)

#checking the model for the tests
model.eval()
with torch.no_grad():
    test_preds = model(X_test_tensor)

#test predictions turned into binary
test_preds_binary = (test_preds > 0.5).float()

accuracy = (test_preds_binary == y_test_tensor).sum().item() / len(y_test_tensor)
print(f"Accuracy: {accuracy * 100:.2f}%")