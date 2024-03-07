import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features=323, out_features=32)
        # self.fc2 = nn.Linear(in_features=32, out_features=16)
        self.fc3 = nn.Linear(in_features=32, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(df_train_encoded, df_test_encoded):
    global percentageOfOnes
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_eval = torch.tensor(df_test_encoded.drop('Label', axis=1).values, dtype=torch.float32)
    X_eval = X_eval.to(device)

    labels = df_train_encoded['Label']
    features = df_train_encoded.drop('Label', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    print(X_train.shape, X_eval.shape)

    model = Model().to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    epochs = 8000
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []

        for train_index, test_index in kf.split(X_train):
            X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
            y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

            y_logits = model(X_train_fold).squeeze()
            y_pred = torch.round(torch.sigmoid(y_logits))
            percentageOfOnes = (y_pred.sum().item() / len(y_pred)) * 100
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
                model.eval()
                y_logits_test = model(X_test_fold).squeeze()
                y_pred_test = torch.round(torch.sigmoid(y_logits_test))
                correct_test = torch.eq(y_pred_test, y_test_fold).sum().item()
                total_test = len(y_test_fold)
                acc_test = (correct_test / total_test) * 100
                loss_test = loss_fn(y_logits_test, y_test_fold)
                mcc_val = matthews_corrcoef(y_test_fold.to('cpu').numpy(), y_pred_test.to('cpu').numpy())

                test_losses.append(loss_test.item())
                test_accuracies.append(acc_test)

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_acc = sum(train_accuracies) / len(train_accuracies)
        avg_test_loss = sum(test_losses) / len(test_losses)
        avg_test_acc = sum(test_accuracies) / len(test_accuracies)

        print(
            f"Epoch [{epoch + 1}/{epochs}], Ones(% of 1): {percentageOfOnes:.4f}, Train Loss: {avg_train_loss:.4f}, Train Accuracy:"
            f"{avg_train_acc:.2f}%, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_acc:.2f}%, MCC: {mcc_val:.4f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time:.2f}s")
    # torch.save(model.state_dict(), 'trained_model.pth')
    return X_eval, df_test_encoded, model


def generate_submission(X_eval, set_to_evaluate, model):
    model.eval()
    with torch.no_grad():
        y_logits_eval = model(X_eval).squeeze()
        y_pred_eval = torch.sigmoid(y_logits_eval)
        y_pred_eval = y_pred_eval.to('cpu')
        set_to_evaluate['Probability_0'] = 1 - y_pred_eval
        set_to_evaluate['Probability_1'] = y_pred_eval
        set_to_evaluate['Label'] = torch.round(y_pred_eval).numpy()
        set_to_evaluate['Label'] = set_to_evaluate['Label'].astype(int)
        set_to_evaluate = set_to_evaluate[['Label', 'Probability_0', 'Probability_1']]
        df = set_to_evaluate[['Label', 'Probability_0', 'Probability_1']]
        df['Probability_0'] = df['Probability_0'].round(2)
        df['Probability_1'] = df['Probability_1'].round(2)
        set_to_evaluate.to_csv('ReAdmitNet_pokusaj3__7_3_2024.csv', index=False)
    print(set_to_evaluate)
    return set_to_evaluate


def train_and_generate_submission(df_train_encoded, df_test_encoded):
    _X_eval, _set_to_evaluate, _model = train_model(df_train_encoded, df_test_encoded)
    return generate_submission(_X_eval, _set_to_evaluate, _model)


if __name__ == "__main__":
    X_eval, set_to_evaluate, model = train_model(
        df_train_encoded = pd.read_csv('./Tim_22/Podaci/train_modified_pesti_encoded.csv'),
        df_test_encoded = pd.read_csv('./Tim_22/Podaci/test_modified_pesti_encoded.csv')
    )
    generate_submission(X_eval, set_to_evaluate, model)
