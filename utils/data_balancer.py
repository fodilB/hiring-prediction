__author__ = 'BENALI Fodil'
__email__ = 'fodel.benali@gmail.com'
__copyright__ = 'Copyright (c) 2023, Technical Test'

import pandas as pd


class DataBalancer:

    def under_sampling(self, data, dependent_variable="embauche", random_state=1):
        df_balanced_1 = data[data[dependent_variable] == 1]

        df_balanced_0 = data[data[dependent_variable] == 0].sample(frac=1, random_state=random_state)[
                        :len(df_balanced_1)]

        data_balanced = pd.concat([df_balanced_0, df_balanced_1])

        return data_balanced
