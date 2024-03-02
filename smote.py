import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Load CSV data
data = pd.read_csv('Tim_22/Podaci/modified_train.csv')

# Initialize the LabelEncoder
le = LabelEncoder()

# Apply encoding to each non-numeric column
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = le.fit_transform(data[column])

data.to_csv('Tim_22/Podaci/tokenized_file.csv', index=False)

print("Data saved to 'tokenized_file.csv'")

data = pd.read_csv('Tim_22/Podaci/tokenized_file.csv')

X = data.drop(columns=['Label'])
y = data['Label']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(y_resampled)
print(X_resampled)

# Drop rows with missing values in the target variable
data.dropna(subset=['Label'], inplace=True)

# Separate features and target variable
X = data.drop(columns=['Label'])
y = data['Label']

# Perform SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Combine resampled features and target variable into a DataFrame
balanced_data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='Label')], axis=1)

# Save balanced dataset to a new CSV file
balanced_data.to_csv('Tim_22/Podaci/balanced_dataset_smote.csv', index=False)

# Visualize class distribution before and after balancing
plt.figure(figsize=(10, 5))

# Plot class distribution before balancing
plt.subplot(1, 2, 1)
plt.bar(data['Label'].value_counts().index, data['Label'].value_counts().values)
plt.title('Class Distribution before Balancing')
plt.xlabel('Class')
plt.ylabel('Count')

# Plot class distribution after balancing
plt.subplot(1, 2, 2)
plt.bar(balanced_data['Label'].value_counts().index, balanced_data['Label'].value_counts().values)
plt.title('Class Distribution after Balancing (SMOTE)')
plt.xlabel('Class')
plt.ylabel('Count')

plt.tight_layout()
plt.show()