__author__ = 'BENALI Fodil'
__email__ = 'fodel.benali@gmail.com'
__copyright__ = 'Copyright (c) 2023, Technical Test'

from sklearn import preprocessing
import pandas as pd


class FeaturesEncoders:

    def one_hot_encoding(self, data, one_hot_encoding_vars):

        for categorical_var in one_hot_encoding_vars:
            # One hot encode the variable
            one_hot_encoded = pd.get_dummies(data[categorical_var], drop_first=True)

            # Concatenate the encoded variable with the original dataset
            data = pd.concat([data, one_hot_encoded], axis=1)

            # Drop the original categorical variable
            data = data.drop(categorical_var, axis=1)

        return data

    def label_encoding_1(self, data, label_encoding_vars):

        for categorical_var in label_encoding_vars:
            # Create a label encoder
            le = preprocessing.LabelEncoder()

            # Fit the label encoder to the categorical variable
            le.fit(data[categorical_var])

            # Transform the categorical variable and store the encoded values in a new column
            data[categorical_var] = le.transform(data[categorical_var])

        return data

    # Manual Label Encoding for ordinal variables
    def label_encoding_2(self, data, feature, encoding_dic):

        data[feature].replace(encoding_dic, inplace=True)

        return data
