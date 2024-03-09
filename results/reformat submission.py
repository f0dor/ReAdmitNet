import pandas as pd


data = pd.read_csv('test/evaluation_results.csv')

df = data[['Label', 'Probability_0', 'Probability_1']]


df['Probability_0'] = df['Probability_0'].round(2)
df['Probability_1'] = df['Probability_1'].round(2)

df.to_csv("ReAdmitNet_2rjesenje_6_3_2024.csv", index=False)


print("done")