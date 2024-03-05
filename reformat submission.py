import pandas as pd


data = pd.read_csv('evaluation_results.csv')

df = data[['Label', 'Probability_0', 'Probability_1']]


df['Probability_0'] = df['Probability_0'].round(2)
df['Probability_1'] = df['Probability_1'].round(2)

df.to_csv("submission_5_3_2024.csv", index=False)


print("done")