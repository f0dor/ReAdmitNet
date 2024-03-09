import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef

# Load the data
data = pd.read_csv('./Tim_22/Podaci/train_modified_encoded1.csv')
set_to_evaluate = pd.read_csv('./Tim_22/Podaci/test_modified_encoded1.csv')

# Split features and labels
X_train, X_test, y_train, y_test = train_test_split(data.drop('Label', axis=1), data['Label'], test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Train the Random Forest classifier
rf_classifier.fit(X_train, y_train)

# Predict probabilities for the evaluation set
eval_probabilities = rf_classifier.predict_proba(set_to_evaluate.drop('Label', axis=1))

# Assign labels based on probabilities
eval_labels = (eval_probabilities[:, 1] >= 0.5).astype(int)

# Calculate Matthews Correlation Coefficient (MCC)
mcc_score = matthews_corrcoef(set_to_evaluate['Label'].values, eval_labels)

print("Matthews Correlation Coefficient:", mcc_score)

# Create a DataFrame to store results
evaluation_results = pd.DataFrame({
    'Label': eval_labels,
    'Probability_0': eval_probabilities[:, 0],
    'Probability_1': eval_probabilities[:, 1]
})

# Save the results to a CSV file
evaluation_results.to_csv('evaluation_results_rf.csv', index=False)
