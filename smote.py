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

# Separate features and target variable
X = data.drop(columns=['Label'])
y = data['Label']

# Define the percentages of minority class to sample
percentages = [10, 15, 20, 25, 100]  # You can adjust these percentages as needed

# Create subplots for displaying the graphs
fig, axs = plt.subplots(len(percentages), figsize=(10, 5*len(percentages)))

# Sample the minority class up to each percentage
for i, percentage in enumerate(percentages):
    # Perform SMOTE to balance the dataset up to the specified percentage
    smote = SMOTE(sampling_strategy=percentage/100, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Combine resampled features and target variable into a DataFrame
    balanced_data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='Label')], axis=1)
    
    # Save balanced dataset to a new CSV file
    balanced_data.to_csv(f'Tim_22/Podaci/balanced_dataset_smote_{percentage}percent.csv', index=False)
    
    # Visualize class distribution
    axs[i].bar(balanced_data['Label'].value_counts().index, balanced_data['Label'].value_counts().values)
    axs[i].set_title(f'Class Distribution after Balancing (SMOTE, {percentage}% Minority)')
    axs[i].set_xlabel('Class')
    axs[i].set_ylabel('Count')
    axs[i].set_xticks(balanced_data['Label'].unique())

# Adjust layout to prevent overlap of subplots
plt.tight_layout()
plt.show()