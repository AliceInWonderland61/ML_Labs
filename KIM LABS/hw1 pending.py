import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import csv
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch.nn as nn
import torch.optim as optim
# For reproducibility
torch.manual_seed(6379)
torch.cuda.manual_seed(6379)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

url_train = "https://hpc.utrgv.edu/static/data/sedan_vs_pickup/train.csv"
train_df = pd.read_csv(url_train)

# Assuming the first and last column are the ID and label
X = train_df.iloc[:, 1:-1].values
y = train_df.iloc[:, -1].values
print(X.shape)
print(y.shape)

# Preprocess the Data
# Normalize features
scaler = MinMaxScaler() #StandarScaler()
X = scaler.fit_transform(X)
print(X.shape)
# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X)
y_train_tensor = torch.LongTensor(y)  # Use LongTensor for classification labels

# Create a Dataset Class
class CustomDataset_Train(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)
#modify
    def __getitem__(self, index):
        return self.features[index], self.labels[index]

# Creating dataset
dataset = CustomDataset_Train(X_train_tensor, y_train_tensor)

# DataLoader
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define the Neural Network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(X.shape[1], 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(64, len(torch.unique(y_train_tensor)))

    def forward(self, x):
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        x = self.dropout3(self.relu3(self.bn3(self.fc3(x))))
        x = self.fc4(x)
        return x

model = SimpleNN().to(device)


# Training Loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


for epoch in range(50):  # epochs
    for inputs, labels in loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
       # print(loss)

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
torch.save(model, 'model.pth')
model = torch.load('model.pth')
url_test = "https://hpc.utrgv.edu/static/data/sedan_vs_pickup/test.csv"
test_df = pd.read_csv(url_test)
X_test = test_df.values
IDs = X_test[:, 0]
X_test_tensor = torch.FloatTensor(X_test[:, 1:])
# Set the model to evaluation mode
model.eval()
predic_corr=0
predictions = []
index = 0
total=0
# Disable gradient computation for efficiency
with torch.no_grad():
    outputs = model(X_test_tensor.to(device))
    _, predicted = torch.max(outputs.data, 1)
    predic_corr+=(predicted.cpu().numpy()==labels).sum().item
    total+=len(labels)
    index += 1
acc=predic_corr/total

predicted=predicted.cpu()
predicted = predicted.numpy()
predictions = np.dstack((IDs, predicted))[0]
with open('submission.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(['ID','Class'])
    write.writerows(predictions)

print(f"Accuracy: {acc *100:.2f}%")
