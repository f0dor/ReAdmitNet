import torch
import torch.nn as nn
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split

def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item() 
  acc = (correct/len(y_pred)) * 100
  return acc

data = pd.read_csv('./Tim_22/Podaci/train_modified_encoded1.csv')

labels = data['Label']
features = data.drop('Label', axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

device = "cuda" if torch.cuda.is_available() else "cpu"

# print(X_train[:5], y_train[:5])
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features = 323,out_features = 512)
        self.fc2 = nn.Linear(in_features = 512, out_features = 128)
        self.fc3 = nn.Linear(in_features = 128, out_features = 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Model().to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

epochs = 10000
for epoch in range(epochs):
    model.train()
    y_logits = model(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    loss = loss_fn(y_logits, y_train)
    correct_train = torch.eq(y_pred, y_train).sum().item()
    total_train = len(y_train)
    acc = (correct_train / total_train) * 100
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        model.eval()
        y_logits_test = model(X_test).squeeze()
        y_pred_test = torch.round(torch.sigmoid(y_logits_test))
        correct_test = torch.eq(y_pred_test, y_test).sum().item()
        total_test = len(y_test)
        acc_test = (correct_test / total_test) * 100
        loss_test = loss_fn(y_logits_test, y_test)
    
    print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%, Test Loss: {loss_test.item():.4f}, Test Accuracy: {acc_test:.2f}%")