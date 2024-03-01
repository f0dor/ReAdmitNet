import pandas as pd

columns_to_encode = ['Age_Group', 'Gender', 'AdmissionType', 'AdmissionDx', 'Dx_Discharge', 'Discharge_Status', 'Education', 'Current_Work_Status']

# Load your CSV file into a DataFrame
df_train = pd.read_csv("./Tim_22/Podaci/train.csv")
df_test = pd.read_csv("./Tim_22/Podaci/test.csv")

# Identify unique categorical classes in the training set
unique_classes_train = {}
for column in columns_to_encode:
    unique_classes_train[column] = df_train[column].unique()

# Identify unique categorical classes in the test set
unique_classes_test = {}
for column in columns_to_encode:
    unique_classes_test[column] = df_test[column].unique()

# Compare unique classes between training and test sets
for column in columns_to_encode:
    train_classes = set(unique_classes_train[column])
    test_classes = set(unique_classes_test[column])
    
    print(f"Column: {column}")
    print("Unique classes in training set:", train_classes)
    print("Unique classes in test set:", test_classes)
    
    # Find the differences between unique classes
    missing_classes = train_classes - test_classes
    extra_classes = test_classes - train_classes
    
    print("Classes missing in test set:", missing_classes)
    print("Extra classes in test set:", extra_classes)
    print()

'''
# Perform one-hot encoding with specified columns
df_encoded_train = pd.get_dummies(df_train, columns=columns_to_encode)
df_encoded_test = pd.get_dummies(df_test, columns=columns_to_encode)

# Save the encoded DataFrame to a new CSV file
df_encoded_train.to_csv("encoded_train_file.csv", index=False)
df_encoded_test.to_csv("encoded_test_file.csv", index=False)

num_features_train = df_encoded_train.shape[1]
num_features_test = df_encoded_test.shape[1]
print("Number of features (columns) in the training set:", num_features_train)
print("Number of features (columns) in the testing set:", num_features_test)
'''