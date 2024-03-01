import pandas as pd
from pandas.io.clipboard import copy


def main():
    data = pd.read_csv("train.csv", header=None)

    for i in range(0, 28588):
        if data[2][i] == "Hitni":
            data[2][i] = 5
        elif data[2][i] == "Elektivni":
            data[2][i] = 0
    col = data.pop(data[0][16])
    data.insert(274, "label", col)
    data.drop(data.columns[15], axis=1, inplace=True)
    data.drop(data.columns[14], axis=1, inplace=True)
    data.drop(data.columns[11], axis=1, inplace=True)
    data.drop(data.columns[10], axis=1, inplace=True)
    data.drop(data.columns[6], axis=1, inplace=True)
    data.drop(data.columns[1], axis=1, inplace=True)
    data.drop(data.columns[0], axis=1, inplace=True)
    data.drop(data.columns[3], axis=1, inplace=True)

    print(data[:][0:10])
    data.to_csv("trainclean.csv")


if __name__ == "__main__":
    main()
