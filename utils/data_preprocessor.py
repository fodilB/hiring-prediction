__author__ = 'BENALI Fodil'
__email__ = 'fodel.benali@gmail.com'
__copyright__ = 'Copyright (c) 2023, Technical Test'

import pandas as pd

import numpy as np


# Data Preparation class
class DataPreprocessor:

    # Load data from a CSV File
    def load_data(self, filepath):
        return pd.read_csv(filepath)

    # Get Details about Data
    def get_data_details(self, df):

        # Get number of columns to have idea about dimensionality to choose the most appropriate algorithms
        print(f"Number of Columns:{df.shape[1]}, number of observations: {df.shape[0]} \n")

        categorial_features = []

        numerical_features = []

        other_features = []

        if ((df.shape[0] > 0) & (df.shape[1] > 0)):

            # Get the data types of the columns
            column_types = df.dtypes

            # Print the column names, data types, proportion of nan values in order to define the most adequate cleaning pipeline
            for col, col_type in column_types.items():

                print(
                    f"Column: {col}, Type: {col_type}, Proportion of Nan values: {round(df[col].isna().sum() / df.shape[0], 3)}%")

                if col_type == 'object':
                    categorial_features.append(col)

                elif col_type in [np.int64, np.float64]:
                    numerical_features.append(col)

                else:
                    other_features.append(col)

            print(f"\n{len(categorial_features)} Categorical Features: {categorial_features}, \
                  \n\n{len(numerical_features)} Numerical Features: {numerical_features},\
                  \n\n{len(other_features)} Other Features: {other_features} \n ")

        return categorial_features, numerical_features, other_features

    def remove_columns(self, data, to_drop):

        # Drop Duplicates Columns
        print(f'--------- . Drop Useless Columns: {to_drop}')
        data.drop(columns=to_drop, inplace=True)

    def clean_data(self, data):

        # Trim whitespace from strings
        print('--------- . Trim whitespace from strings')
        data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        # Drop Duplicates Rows
        print('--------- . Drop Duplicates Rows')
        data.drop_duplicates(inplace=True)

        # Remove rows with missing values :
        # We have noticed that there is not a lot of missing values, for that we prefer remove its from the dataset
        print('--------- . Remove rows with missing values')
        data.dropna(inplace=True)

        return data
