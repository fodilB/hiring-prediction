__author__ = 'BENALI Fodil'
__email__ = 'fodel.benali@gmail.com'
__copyright__ = 'Copyright (c) 2023, Technical Test'

import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from scipy.stats import ks_2samp
from scipy.stats import ttest_ind


class HypothesisTestEvaluation:

    # Evaluate  association between pair of variables using Chi2 test
    def chi2_test(self, data, col1, col2="embauche"):

        # Create a contingency table
        cont_table = pd.crosstab(data[col1], data[col2])

        # Perform the chi-square test
        chi2, p, dof, ex = chi2_contingency(cont_table)

        # Determine the test result
        if p < 0.05:
            print(f"p-value={p} < 0.05 ==>  Significant Dependence using Chi2 Test")
        else:
            print(f"p-value={p} >= 0.05 ==> No Significant Dependence using Chi2 Test")

    # Compare distribution using two-sample Kolmogorov-Smirnov test
    def compare_distribution_using_ks_test(self, data, col1, col2="embauche"):

        statistic, p = ks_2samp(data[data[col2] == 1][col1].values, data[data[col2] == 1][col1].values)

        # Determine the test result
        if p < 0.05:
            print(
                f"p-value={p} < 0.05 ==>  Significant Dependence using Chi2 Test (significant difference between distributions)")
        else:
            print(
                f"p-value={p} >= 0.05 ==> No Significant Dependence using two-sample Kolmogorov-Smirnov test (same distribution)")

    # Compare distribution using  two-sample t-test
    def compare_distribution_using_tt_test(self, data, col1, col2="embauche"):

        stat, p = ttest_ind(data[data[col2] == 1][col1].values, data[data[col2] == 1][col1].values)

        # Determine the test result
        if p < 0.05:
            print(
                f"p-value={p} < 0.05 ==>  Significant Dependence using T Test (significant difference between distributions means)")
        else:
            print(
                f"p-value={p} >= 0.05 ==> No Significant Dependence using T test (no significant difference between distributions means)")

    def assess_dependence_using_anova_test(self, data, cat_col, cont_col):

        # Group the continuous column by the categorical column
        grouped_df = data.groupby(cat_col)[cont_col]

        # Perform the one-way ANOVA test
        statistic, pvalue = f_oneway(*[grouped_df.get_group(x) for x in grouped_df.groups])

        p = pvalue

        # Determine the test result
        if p < 0.05:
            print(f"p-value={p} < 0.05 ==>  Significant Dependence using Anova one way Test")
        else:
            print(f"p-value={p} >= 0.05 ==> No Significant Dependence using Anova one way Test")
