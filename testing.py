import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Load first SMOTE-balanced dataset for training
balanced_data_train = pd.read_csv('Tim_22/Podaci/balanced_dataset_smote.csv')

# Separate features and target variable in the training dataset
X_train = balanced_data_train.drop(columns=['Label'])
y_train = balanced_data_train['Label']

# Initialize and train Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Load second CSV file for prediction
data_to_predict = pd.read_csv('Tim_22/Podaci/test.csv')

# Preprocess the data from the second CSV file (if necessary)
# Ensure that it has the same features as the training data
le = LabelEncoder()

for column in data_to_predict.columns:
    if data_to_predict[column].dtype == 'object':
        data_to_predict[column] = le.fit_transform(data_to_predict[column])

# Extract features for prediction
X_to_predict = data_to_predict.drop(columns=['Label', 'Probability_0', 'Probability_1'])

# Predict the probabilities for each class (0 or 1) for the other two columns
probabilities = rf_classifier.predict_proba(X_to_predict)

# Add predicted probabilities to the original DataFrame
data_to_predict['Probability_0'] = probabilities[:, 0]
data_to_predict['Probability_1'] = probabilities[:, 1]

predicted_values = [1 if prob[1] > 0.5 else 0 for prob in probabilities]

# Add predicted values to the original DataFrame
data_to_predict['Label'] = predicted_values

# Save the DataFrame with predicted probabilities to a new CSV file
data_to_predict.to_csv('Tim_22/Podaci/predicted_data.csv', index=False)