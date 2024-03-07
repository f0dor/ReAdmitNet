import torch
import torch.nn as nn
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from sklearn.model_selection import KFold

device = "cuda" if torch.cuda.is_available() else "cpu"

data = pd.read_csv('./Tim_22/Podaci/train_modified_encoded1.csv')
set_to_evaluate = pd.read_csv('./Tim_22/Podaci/test_modified_encoded1.csv')

X_eval = torch.tensor(set_to_evaluate.drop('Label', axis=1).values, dtype=torch.float32)
X_eval = X_eval.to(device)


# Separate features and target variable
X = data.drop(columns=['Label'])
y = data['Label']

# Define your model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features=323, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)  # Define dropout layer with dropout probability

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after the activation function
        x = self.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout after the activation function
        x = self.fc3(x)
        return x

# Initialize your model and move it to the appropriate device
model = Model().to(device)

# Define your loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Convert your data to tensors and move them to the appropriate device
X = torch.tensor(X.values, dtype=torch.float32).to(device)
y = torch.tensor(y.values, dtype=torch.float32).to(device)

# Define your training and testing loops
epochs = 5000
kf = KFold(n_splits=5, shuffle=True, random_state=42)
start_time = time.time()

for epoch in range(epochs):
    model.train()  # Set the model to training mode
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    for train_index, test_index in kf.split(X):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        
        y_logits = model(X_train_fold).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))
        loss = loss_fn(y_logits, y_train_fold)
        correct_train = torch.eq(y_pred, y_train_fold).sum().item()
        total_train = len(y_train_fold)
        acc = (correct_train / total_train) * 100
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        train_accuracies.append(acc)
        
        with torch.no_grad():
            model.eval()  # Set the model to evaluation mode
            y_logits_test = model(X_test_fold).squeeze()
            y_pred_test = torch.round(torch.sigmoid(y_logits_test))
            correct_test = torch.eq(y_pred_test, y_test_fold).sum().item()
            total_test = len(y_test_fold)
            acc_test = (correct_test / total_test) * 100
            loss_test = loss_fn(y_logits_test, y_test_fold)
            
            test_losses.append(loss_test.item())
            test_accuracies.append(acc_test)
    
    avg_train_loss = sum(train_losses) / len(train_losses)
    avg_train_acc = sum(train_accuracies) / len(train_accuracies)
    avg_test_loss = sum(test_losses) / len(test_losses)
    avg_test_acc = sum(test_accuracies) / len(test_accuracies)
    
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_acc:.2f}%, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_acc:.2f}%")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training time: {elapsed_time:.2f}s")

# Save your model if needed
# torch.save(model.state_dict(), 'trained_model.pth')

# Load the evaluation set
X_eval = torch.tensor(set_to_evaluate.drop('Label', axis=1).values, dtype=torch.float32).to(device)

# Perform inference on the evaluation set
model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    y_logits_eval = model(X_eval).squeeze()
    y_pred_eval = torch.sigmoid(y_logits_eval)
    y_pred_eval = y_pred_eval.to('cpu')
    set_to_evaluate['Probability_0'] = 1 - y_pred_eval
    set_to_evaluate['Probability_1'] = y_pred_eval
    set_to_evaluate['Label'] = torch.round(y_pred_eval).cpu().numpy().astype(int)
    set_to_evaluate = set_to_evaluate[['Label', 'Probability_0', 'Probability_1']]
    set_to_evaluate.to_csv('evaluation_results.csv', index=False)