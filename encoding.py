import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

## Promjenit ovu liniju za neki drugi csv file
train_path = "./Tim_22/Podaci/train_modified.csv"
test_path = "./Tim_22/Podaci/test_modified.csv"
savepath_train = train_path.split(".csv")[0] + "_encoded.csv"
savepath_test = test_path.split(".csv")[0] + "_encoded.csv"

def main():
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_train = df_train.iloc[:, 1:]
    df_test = df_test.iloc[:, 1:-2]
    columns = df_test.columns.tolist()
    
    data_types = df_train.dtypes

    categorical_columns = [column for column, data_type in data_types.items() if data_type == 'object']
    unique_values_train = [df_train[column].unique().tolist() for column in categorical_columns]
    unique_values_test = [df_test[column].unique().tolist() for column in categorical_columns]

    different_values = []
    for column in categorical_columns:
        train_values = set(unique_values_train[categorical_columns.index(column)])
        test_values = set(unique_values_test[categorical_columns.index(column)])
        if train_values != test_values:
            different_values.extend(list(train_values.symmetric_difference(test_values)))
    
    for value in different_values:
        df_test[value] = 0

    # Iterate over each column
    for column in columns:
        # Check the data type of the column
        data_type = data_types[column]

        # Apply the appropriate encoding method based on the data type
        if data_type == 'object':
            # Categorical data - use OneHotEncoder
            encoder = OneHotEncoder()
            encoded_data_train = encoder.fit_transform(df_train[[column]])
            encoded_data_test = encoder.transform(df_test[[column]])
            df_encoded_train = pd.DataFrame(encoded_data_train.toarray(), columns=encoder.get_feature_names_out([column]))
            df_encoded_test = pd.DataFrame(encoded_data_test.toarray(), columns=encoder.get_feature_names_out([column]))
            df_train = pd.concat([df_train, df_encoded_train], axis=1)
            df_test = pd.concat([df_test, df_encoded_test], axis=1)
            df_train = df_train.drop(column, axis=1)
            df_test = df_test.drop(column, axis=1)
        elif data_type == 'int64' or data_type == 'float64' or data_type == 'float32' or data_type == 'int32':
            # Numeric data - use StandardScaler
            scaler = StandardScaler()
            scaled_data_train = scaler.fit_transform(df_train[[column]])
            scaled_data_test = scaler.transform(df_test[[column]])
            df_train[column] = scaled_data_train
            df_test[column] = scaled_data_test

    # Save the encoded dataframe to a new CSV file
    df_train.to_csv(savepath_train, index=False)
    df_test.to_csv(savepath_test, index=False)

if __name__ == "__main__":
    main()