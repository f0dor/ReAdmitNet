import pandas as pd

# Load the two CSV files
file1 = pd.read_csv('./results/ReAdmitNet_pokusaj4__2024_03_08_17_27_50.csv')
file2 = pd.read_csv('./results/ReAdmitNet_pokusaj4__2024_03_08_17_36_06.csv')

# Count the number of ones in each file
ones_file1 = (file1['Label'] == 1).sum()
ones_file2 = (file2['Label'] == 1).sum()

# Count the number of matching ones
matching_ones = ((file1['Label'] == 1) & (file2['Label'] == 1)).sum()

print("Number of ones in file 1:", ones_file1)
print("Number of ones in file 2:", ones_file2)
print("Number of matching ones:", matching_ones)

# Calculate the percentage of ones in each file
total_samples_file1 = len(file1)
total_samples_file2 = len(file2)

percentage_ones_file1 = (file1['Label'] == 1).mean() * 100
percentage_ones_file2 = (file2['Label'] == 1).mean() * 100

print("Percentage of ones in file 1: {:.2f}%".format(percentage_ones_file1))
print("Percentage of ones in file 2: {:.2f}%".format(percentage_ones_file2))
