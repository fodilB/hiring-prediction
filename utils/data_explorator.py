__author__ = 'BENALI Fodil'
__email__ = 'fodel.benali@gmail.com'
__copyright__ = 'Copyright (c) 2023, Technical Test'

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
plt.rc("font", size=14)


class DataExplorator:

    def dependent_variable_proportions(self, data, dependent_variable='embauche'):

        sns.countplot(x=dependent_variable, data=data, palette='hls').set_title('Proportion d\'embauche')

        plt.show()

        proportion = data[dependent_variable].sum() / data.shape[0]

        print(f"Pourcentage d'embauche : {proportion * 100:.2f} %\n")

        return proportion

    def feature_proportions_per_dependent_variable(self, data, feature, dependent_variable='embauche'):
        ax = sns.countplot(x=feature, data=data, palette='hls', hue=dependent_variable).set_title(
            f'Proportion d\'embauche par {feature}')

        plt.show()

        freq = {}

        for label in data[feature].unique():
            freq[label] = str(round(
                100 * data[feature][(data[feature] == label) & (data[dependent_variable] == 1)].count() / data[feature][
                    data[feature] == label].count(), 3)) + ' %'

        print(f'Hiring Proportion per category:{freq}\n')

    def feature_distribution_per_dependent_variable(self, data, feature, dependent_variable='embauche'):
        ax = sns.displot(data=data, x=feature, hue=dependent_variable, kind="kde")

        ax.figure.suptitle(f'Distribution of hiring per obtained {feature}')

        plt.show()
