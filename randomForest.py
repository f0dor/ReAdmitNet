from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

# Load CSV data
#data = pd.read_csv('Tim_22/Podaci/tokenized_file.csv')
data = pd.read_csv('Tim_22/Podaci/predicted_data.csv')

# Separate features and target variable
X = data.drop(columns=['Label'])
y = data['Label']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate class weights
class_weights = dict(zip([0, 1], (len(y_train) / (2 * np.bincount(y_train)))))

# Initialize and train Random Forest classifier with class weights
rf_classifier = RandomForestClassifier(class_weight=class_weights, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate classifier performance
print(classification_report(y_test, y_pred))
