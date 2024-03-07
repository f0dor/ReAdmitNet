import pandas as pd

df_train = pd.read_csv("../Tim_22/Podaci/train.csv")
df_test = pd.read_csv("../Tim_22/Podaci/test.csv")

column_names = df_train.columns.tolist()


### Making age a single number, replacing missing values with the average

def calculate_mean(age_range):
    start, end = map(int, age_range.split('-'))
    mean_age = (start + end) / 2
    return mean_age


df_train['Age_Group'] = df_train['Age_Group'].apply(calculate_mean).astype('float32')
df_test['Age_Group'] = df_test['Age_Group'].apply(calculate_mean).astype('float32')

## Replace missing values with average age
average_age_train = df_train['Age_Group'].mean()
average_age_test = df_test['Age_Group'].mean()

df_train['Age_Group'] = df_train['Age_Group'].fillna(average_age_train).astype('float32')
df_test['Age_Group'] = df_test['Age_Group'].fillna(average_age_test).astype('float32')


### PreviousAdmissionDays changing -8 to 0, replacing missing values with the average

def replace_and_convert_train(value):
    if pd.isna(value) or value < 0:
        return 0.0
    else:
        return float(value)


def replace_and_convert_test(value):
    if pd.isna(value) or value < 0:
        return 0.0
    else:
        return float(value)


df_train['PreviousAdmissionDays'] = df_train['PreviousAdmissionDays'].apply(replace_and_convert_train).astype('float32')
df_test['PreviousAdmissionDays'] = df_test['PreviousAdmissionDays'].apply(replace_and_convert_test).astype('float32')

### LOS and LOS_ICU to float, replacing missing values with the average

df_train['LOS'] = df_train['LOS'].fillna(df_train['LOS'].mean()).astype('float32')
df_train['LOS_ICU'] = df_train['LOS_ICU'].fillna(df_train['LOS_ICU'].mean()).astype('float32')
df_test['LOS'] = df_test['LOS'].fillna(df_test['LOS'].mean()).astype('float32')
df_test['LOS_ICU'] = df_test['LOS_ICU'].fillna(df_test['LOS_ICU'].mean()).astype('float32')

### Surgery_Count to float

df_train['Surgery_Count'] = df_train['Surgery_Count'].astype('float32')
df_test['Surgery_Count'] = df_test['Surgery_Count'].astype('float32')

### Discharge_Specialty to float

df_train['Discharge_Specialty'] = df_train['Discharge_Specialty'].astype('float32')
df_test['Discharge_Specialty'] = df_test['Discharge_Specialty'].astype('float32')

### Weight_Discharge and Height_Discharge to float32, replacing missing and negative values with average regards to gender

## Calculating average weights for males and females in both training and test sets

male_rows_train = df_train[df_train['Gender'] == 'M']
male_weights_train = male_rows_train['Weight_Discharge']
average_male_weight_train = male_weights_train.mean()

male_rows_test = df_test[df_test['Gender'] == 'M']
male_weights_test = male_rows_test['Weight_Discharge']
average_male_weight_test = male_weights_test.mean()

female_rows_train = df_train[df_train['Gender'] == 'Ž']
female_weights_train = female_rows_train['Weight_Discharge']
average_female_weight_train = female_weights_train.mean()

female_rows_test = df_test[df_test['Gender'] == 'Ž']
female_weights_test = female_rows_test['Weight_Discharge']
average_female_weight_test = female_weights_test.mean()

## Calculating average heights for males and females in both training and test sets

male_heights_train = male_rows_train['Height_Discharge']
average_male_height_train = male_heights_train.mean()

male_heights_test = male_rows_test['Height_Discharge']
average_male_height_test = male_heights_test.mean()

female_heights_train = female_rows_train['Height_Discharge']
average_female_height_train = female_heights_train.mean()

female_heights_test = female_rows_test['Height_Discharge']
average_female_height_test = female_heights_test.mean()


## Replacing missing and negative values with average weights and heights

def replace_weight_train(row):
    if row['Weight_Discharge'] <= 0 or pd.isna(row['Weight_Discharge']):
        if row['Gender'] == 'M':
            return average_male_weight_train
        else:
            return average_female_weight_train
    else:
        return row['Weight_Discharge']


def replace_weight_test(row):
    if row['Weight_Discharge'] <= 0 or pd.isna(row['Weight_Discharge']):
        if row['Gender'] == 'M':
            return average_male_weight_test
        else:
            return average_female_weight_test
    else:
        return row['Weight_Discharge']


def replace_height_train(row):
    if row['Height_Discharge'] <= 0 or pd.isna(row['Height_Discharge']):
        if row['Gender'] == 'M':
            return average_male_height_train
        else:
            return average_female_height_train
    else:
        return row['Height_Discharge']


def replace_height_test(row):
    if row['Height_Discharge'] <= 0 or pd.isna(row['Height_Discharge']):
        if row['Gender'] == 'M':
            return average_male_height_test
        else:
            return average_female_height_test
    else:
        return row['Height_Discharge']


df_train['Weight_Discharge'] = df_train.apply(replace_weight_train, axis=1).astype('float32')
df_test['Weight_Discharge'] = df_test.apply(replace_weight_test, axis=1).astype('float32')
df_train['Height_Discharge'] = df_train.apply(replace_height_train, axis=1).astype('float32')
df_test['Height_Discharge'] = df_test.apply(replace_height_test, axis=1).astype('float32')


### Gender, replacing missing values with the mode
def replace_gender(row):
    if pd.isna(row['Gender']):
        if abs(row['Weight_Discharge'] - average_male_weight_train) < abs(
                row['Weight_Discharge'] - average_female_weight_train):
            return 'M'
        else:
            return 'Ž'
    else:
        return row['Gender']


df_train['Gender'] = df_train.apply(replace_gender, axis=1)
df_test['Gender'] = df_test.apply(replace_gender, axis=1)

### Replacing missing categorical values

## AdmissionDx

mode_admissiondx = df_train['AdmissionDx'].mode()[0]
df_train['AdmissionDx'].fillna(mode_admissiondx, inplace=True)
df_test['AdmissionDx'].fillna(mode_admissiondx, inplace=True)

## AdmissionType

mode_admissiontype = df_train['AdmissionType'].mode()[0]
df_train['AdmissionType'].fillna(mode_admissiontype, inplace=True)
df_test['AdmissionType'].fillna(mode_admissiontype, inplace=True)

## Dx_Discharge

mode_dx_discharge = df_train['Dx_Discharge'].mode()[0]
df_train['Dx_Discharge'].fillna(mode_dx_discharge, inplace=True)
df_test['Dx_Discharge'].fillna(mode_dx_discharge, inplace=True)

## Discharge_Status

mode_discharge_status = df_train['Discharge_Status'].mode()[0]
df_train['Discharge_Status'].fillna(mode_discharge_status, inplace=True)
df_test['Discharge_Status'].fillna(mode_discharge_status, inplace=True)

## Education

mode_education = df_train['Education'].mode()[0]
df_train['Education'].fillna(mode_education, inplace=True)
df_test['Education'].fillna(mode_education, inplace=True)

## Current_Work_Status

mode_work_status = df_train['Current_Work_Status'].mode()[0]


def replace_missing(row):
    age = row['Age_Group']
    work_status = row['Current_Work_Status']

    if pd.notnull(age) and age > 65:
        return "UMIROVLJENIK"
    elif pd.isnull(work_status):
        return mode_work_status
    else:
        return work_status


df_train['Current_Work_Status'] = df_train.apply(replace_missing, axis=1)
df_test['Current_Work_Status'] = df_test.apply(replace_missing, axis=1)

df_train.to_csv("./Tim_22/Podaci/modified_train.csv", index=False)
df_test.to_csv("./Tim_22/Podaci/modified_test.csv", index=False)