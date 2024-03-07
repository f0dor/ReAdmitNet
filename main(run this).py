import pandas as pd

from data_prep_refactor import data_prep
from encoding import encoding
from model import train_and_generate_submission


def main():
    originalpathtrain = "./Tim_22/Podaci/train.csv"
    originalpathtest = "./Tim_22/Podaci/test.csv"

    df_train, _ = data_prep(pd.read_csv(originalpathtrain), originalpathtrain)
    df_test, _ = data_prep(pd.read_csv(originalpathtest), originalpathtest)
    df_train_encoded, df_test_encoded = encoding(df_train, df_test)
    train_and_generate_submission(df_train_encoded, df_test_encoded)


if __name__ == "__main__":
    main()
