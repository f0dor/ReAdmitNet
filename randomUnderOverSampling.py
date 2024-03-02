import pandas as pd
from sklearn.utils import resample
import matplotlib.pyplot as plt

# Load CSV data
data = pd.read_csv('Tim_22/Podaci/tokenized_file.csv')

# Perform random undersampling or oversampling
# Assuming 'Label' is the target variable
minority_class = data[data['Label'] == 1]
majority_class = data[data['Label'] == 0]

# Visualize class distribution before resampling
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(['Yes', 'No'], [len(minority_class), len(majority_class)])
plt.title('Class Distribution before Resampling')
plt.xlabel('Class')
plt.ylabel('Count')
# Add the value counts to the bars
for index, value in enumerate([len(minority_class), len(majority_class)]):
    plt.text(index, value, str(value))

# Choose the resampling strategy based on class distribution
if len(minority_class) < len(majority_class):
    # Perform random oversampling of the minority class
    minority_oversampled = resample(minority_class, 
                                    replace=True, 
                                    n_samples=len(majority_class), 
                                    random_state=42)
    # Combine oversampled minority class with majority class
    balanced_data = pd.concat([majority_class, minority_oversampled])
else:
    # Perform random undersampling of the majority class
    majority_undersampled = resample(majority_class, 
                                     replace=False, 
                                     n_samples=len(minority_class), 
                                     random_state=42)
    # Combine undersampled majority class with minority class
    balanced_data = pd.concat([minority_class, majority_undersampled])

plt.subplot(1, 2, 2)
plt.bar(['Yes', 'No'], balanced_data['Label'].value_counts())
plt.title('Class Distribution after Resampling')
plt.xlabel('Class')
plt.ylabel('Count')
plt.tight_layout()
plt.show()