import pandas as pd

## Promjenit ovu liniju za neki drugi csv file
originalpath = "./Tim_22/Podaci/test.csv"

savepath = originalpath.split(".csv")[0] + "_modified.csv"


def main():
    df = pd.read_csv(originalpath)

    #column_names = df.columns.tolist()

    df['Age_Group'] = df['Age_Group'].apply(calculate_mean).astype('float32')

    average_age = df['Age_Group'].mean()
    df['Age_Group'] = df['Age_Group'].fillna(average_age).astype('float32')

    df['PreviousAdmissionDays'] = df['PreviousAdmissionDays'].apply(replace_and_convert).astype('float32')

    df['LOS'] = df['LOS'].fillna(df['LOS'].mean()).astype('float32')
    df['LOS_ICU'] = df['LOS_ICU'].fillna(df['LOS_ICU'].mean()).astype('float32')

    df['Surgery_Count'] = df['Surgery_Count'].astype('float32')

    df['Discharge_Specialty'] = df['Discharge_Specialty'].astype('float32')

    male_rows = df[df['Gender'] == 'M']
    male_weights = male_rows['Weight_Discharge']
    average_male_weight = male_weights.mean()

    female_rows = df[df['Gender'] == 'Ž']
    female_weights = female_rows['Weight_Discharge']
    average_female_weight = female_weights.mean()

    male_heights = male_rows['Height_Discharge']
    average_male_height = male_heights.mean()

    female_heights = female_rows['Height_Discharge']
    average_female_height = female_heights.mean()

    df['Weight_Discharge'] = df.apply(replace_weight, args=(average_male_weight, average_female_weight), axis=1).astype(
        'float32')
    df['Height_Discharge'] = df.apply(replace_height, args=(average_male_height, average_female_height), axis=1).astype(
        'float32')
    df['Gender'] = df.apply(replace_gender, args=(average_male_weight, average_female_weight), axis=1)

    mode_admissiondx = df['AdmissionDx'].mode()[0]
    mode_admissiontype = df['AdmissionType'].mode()[0]
    mode_dx_discharge = df['Dx_Discharge'].mode()[0]
    mode_discharge_status = df['Discharge_Status'].mode()[0]
    mode_education = df['Education'].mode()[0]
    mode_work_status = df['Current_Work_Status'].mode()[0]

    df['AdmissionDx'] = df['AdmissionDx'].fillna(mode_admissiondx)
    df['AdmissionType'] = df['AdmissionType'].fillna(mode_admissiontype)
    df['Dx_Discharge'] = df['Dx_Discharge'].fillna(mode_dx_discharge)
    df['Discharge_Status'] = df['Discharge_Status'].fillna(mode_discharge_status)
    df['Education'] = df['Education'].fillna(mode_education)
    df['Current_Work_Status'] = df.apply(replace_missing, args=(mode_work_status,), axis=1)

    df = remove_trailing_whitespace(df)

    df.to_csv(savepath, index=False)


def replace_missing(row, mode_work_status):
    age = row['Age_Group']
    work_status = row['Current_Work_Status']

    if pd.notnull(age) and age > 65:
        return "UMIROVLJENIK"
    elif pd.isnull(work_status):
        return mode_work_status
    else:
        return work_status


def calculate_mean(age_range):
    start, end = map(int, age_range.split('-'))
    mean_age = (start + end) / 2
    return mean_age


def replace_and_convert(value):
    if pd.isna(value) or value < 0:
        return 0.0
    else:
        return float(value)


def replace_weight(row, average_male_weight, average_female_weight):
    if row['Weight_Discharge'] <= 0 or pd.isna(row['Weight_Discharge']):
        if row['Gender'] == 'M':
            return average_male_weight
        else:
            return average_female_weight
    else:
        return row['Weight_Discharge']


def replace_height(row, average_male_height, average_female_height):
    if row['Height_Discharge'] <= 0 or pd.isna(row['Height_Discharge']):
        if row['Gender'] == 'M':
            return average_male_height
        else:
            return average_female_height
    else:
        return row['Height_Discharge']


def replace_gender(row, average_male_weight, average_female_weight):
    if pd.isna(row['Gender']):
        if abs(row['Weight_Discharge'] - average_male_weight) < abs(
                row['Weight_Discharge'] - average_female_weight):
            return 'M'
        else:
            return 'Ž'
    else:
        return row['Gender']

def remove_trailing_whitespace(df):
    object_columns = df.select_dtypes(include=['object']).columns
    df[object_columns] = df[object_columns].apply(lambda x: x.str.rstrip() if x.dtype == 'object' else x)
    return df


if __name__ == "__main__":
    main()
