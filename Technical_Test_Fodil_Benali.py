#!/usr/bin/env python
# coding: utf-8
__author__ = 'BENALI Fodil'
__email__ = 'fodel.benali@gmail.com'
__copyright__ = 'Copyright (c) 2023, Technical Test'

import sys

import pandas as pd
import numpy as np

from utils.data_balancer import DataBalancer
from utils.data_explorator import DataExplorator
from utils.data_preprocessor import DataPreprocessor
from utils.features_encoders import FeaturesEncoders
from utils.hypothesis_tester import HypothesisTestEvaluation
from utils.model_fitter import ModelFitter

import matplotlib.pyplot as plt
import seaborn as sns

plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

if __name__ == '__main__':
    # Check python version
    print(sys.version)

    # # Question 1 : Data Description
    data_preprocessor = DataPreprocessor()

    print('------------------------------------------ Data structure ---------------------------------------------- \n')

    # Load the data
    file_path = './data.csv'
    data = data_preprocessor.load_data(file_path)
    print(data.head)

    print(
        '\n --------------------------------------------- Data Description ------------------------------------------------- \n')
    # To get infos about  columns
    categorial_features, numerical_featues, other_features = data_preprocessor.get_data_details(data)

    print(
        '\n --------------------------------------------- Data Cleaning ---------------------------------------------\n')

    # Convert columns to the correct data type
    print('--------- . Convert Data To Correct Data Type')

    #  Using errors=’coerce’. It will replace all non-numeric values with NaN/Nat
    data["age"] = pd.to_numeric(data["age"], downcast='integer', errors='coerce')
    data["exp"] = pd.to_numeric(data["exp"], downcast='integer', errors='coerce')
    data["date"] = pd.to_datetime(data["date"], format='%Y-%m-%d', errors='coerce')

    # Filter incoherent data:
    print('--------- . Filter incoherent data (e.g., age < 16, experience<0, note > 100)')
    # You can't graduate before age 16, with a few exceptions,
    # you can't have negative experience and a grade above 100
    data = data[(data["age"] >= 16) & (data["exp"] >= 0) & (data["note"] <= 100) & (data["note"] >= 0)]

    # Drop Useless Columns: Unnamed: 0, index: Duplicated
    # 'cheveux': Discrimination based on physical appearance Not allowed in France, but we will keep it for analysis

    to_drop = ['Unnamed: 0', 'index']
    data_preprocessor.remove_columns(data, to_drop)

    data = data_preprocessor.clean_data(data)

    print(
        '\n --------------------------------------------- Data After Cleaning -----------------------------------------------------\n')
    categorical_features, numerical_featues, other_features = data_preprocessor.get_data_details(data)

    # # Data Exploration
    data_explorator = DataExplorator()

    # <div class="alert alert-block alert-info">
    # The <b>Chi-Squared test</b> is a statistical test used to assess the statistical significance of the association between two categorical variables. It is based on the Chi-Squared statistic, which measures the difference between the observed frequencies and the expected frequencies in the contingency table. The Chi-Squared test is commonly used to determine whether there is a significant association between two categorical variables.
    #
    # If the p-value is small (typically less than 0.05), it indicates that the observed frequencies are significantly different from the expected frequencies, and therefore there is a significant association between the two variables. If the p-value is large, it indicates that the observed frequencies are not significantly different from the expected frequencies, and therefore there is no significant association between the variables.
    # </div>

    # <div class="alert alert-block alert-info">
    #     The <b>T-test</b> is a statistical test used to determine whether there is a significant difference between the means of two groups. It is commonly used to compare the means of two populations. The paired t-test is used to compare the means of two related samples, such as the scores of the same group of people on two different tests. It is used to test the hypothesis that the mean difference between the two tests is zero.
    #
    # To perform a t-test, you need to compute the t-statistic, which is a measure of the difference between the means of the two groups. The t-statistic is calculated as the difference between the means divided by the standard error of the difference. The p-value of the t-test is the probability of obtaining a t-statistic at least as large as the observed value, given that the null hypothesis (no difference between the means) is true.
    #
    # If the p-value is small (typically less than 0.05), it indicates that the observed difference between the means is statistically significant, and therefore there is a significant difference between the two groups. If the p-value is large, it indicates that the observed difference is not statistically significant, and therefore there is no significant difference between the groups.
    # </div>

    # <div class="alert alert-block alert-info">
    # <b>ANOVA (Analysis of Variance)</b> is a statistical test used to determine whether there is a significant difference between the means of two or more groups. It is used to compare the means of multiple groups, and to test the hypothesis that the means of the groups are equal. If the p-value is small (typically less than 0.05), it indicates that the observed difference between the means is statistically significant, and therefore there is a significant difference between the groups. If the p-value is large, it indicates that the observed difference is not statistically significant, and therefore there is no significant difference between the groups.
    #
    # - One-way ANOVA is a statistical test that can be used to evaluate the dependence between a categorical and a continuous variable. The categorical variable is often referred to as the independent variable, while the continuous variable is referred to as the dependent variable.
    #
    # - One-way ANOVA is used to test the null hypothesis that the means of the dependent variable are equal across all levels of the independent variable. If the null hypothesis is rejected, it suggests that there is a significant difference in the means of the dependent variable between at least two levels of the independent variable, indicating a dependence between the two variables.
    #
    # - One-way ANOVA is an appropriate test to use when you have one categorical independent variable and a continuous dependent variable, and you want to test whether the mean of the dependent variable is the same across all levels of the independent variable.
    # </div>

    hypothesis_test_evaluation = HypothesisTestEvaluation()
    print(
        ' ----------------------------------------------- Data Exploration -----------------------------------------------------')

    dependent_variable = 'embauche'

    proportion = data_explorator.dependent_variable_proportions(data)

    if proportion < 0.5:
        print("----------------- Conclusion: Data set is unbalanced ----------------------")

    print(f'Dependence Evaluation between: {categorical_features} and the dependent feature: {dependent_variable}')

    for feature in categorical_features:
        data_explorator.feature_proportions_per_dependent_variable(data, feature)

        hypothesis_test_evaluation.chi2_test(data, feature)

    # #### From these figures, we notice:
    # <div class="alert alert-block alert-success">
    #
    #  - There is <b>not a significant difference in the proportion of hires by gender and availability</b>.
    #
    #  - For the rest of the features, there are a significant differences.
    #
    #  => These findings have been validated by the chi2 dependency test.
    #
    # Therefore, we can assume that there is not enough evidence to consider the gender and availability features as relevant during the hiring process.
    #
    # In the following, we will test some hypotheses that are often made during the recruitment process:
    #  - Are applicants with high grades more likely to be hired?
    #  - Are candidates with more experience more favorable during the recruitment process?
    #  - Can age be a criterion for selecting a candidate?
    #  - Does the year, month, season (e.g., heat in the summer, rain in the winter) impact the hiring process?
    # </div>
    data_explorator.feature_distribution_per_dependent_variable(data, "note")
    hypothesis_test_evaluation.compare_distribution_using_tt_test(data, "note")

    # <div class="alert alert-block alert-success">
    # From the figure we can see that there are several candidates who had very good grades but were not hired. Therefore, <b>this may indicate that having very good grades is not enough to succeed in a recruitment process. This finding have been validated using the T-Test.</b>
    #  </div>
    data_explorator.feature_distribution_per_dependent_variable(data, "age")
    hypothesis_test_evaluation.chi2_test(data, "age")

    # <div class="alert alert-block alert-success">
    # From the figure we can see that there are several candidates who had same age but were not hired. Therefore, <b>this may indicate that the age alone is not a decisive creterion in a recruitment process. This finding have been validated using the Chi2-Test which compares the two distribution.</b>
    #  </div>

    # # Extract features from date attributes: Month, Year, Season
    print(
        '---------------------------------  Extract Features from date attribute -----------------------------------------------')

    data["month"] = data["date"].dt.month.astype('category')

    data["year"] = data["date"].dt.year.astype('category')

    # Create a new column with the saison to characterize the period of hiring the most important
    data["season"] = "Unknown"
    data.loc[data["month"].isin([12, 1, 2]), "season"] = "winter"
    data.loc[data["month"].isin([3, 4, 5]), "season"] = "spring"
    data.loc[data["month"].isin([6, 7, 8]), "season"] = "summer"
    data.loc[data["month"].isin([9, 10, 11]), "season"] = "autumn"
    data["season"].astype('category')

    date_features = ['season', 'year', 'month']

    for feature in date_features:
        data_explorator.feature_proportions_per_dependent_variable(data, feature)

        hypothesis_test_evaluation.chi2_test(data, feature)

    # <div class="alert alert-block alert-success">
    # From these figures and results, we notice that there <b>is no significant difference in the proportions of hiring by season and month</b>. In conclusion, this may lead us to discard date modeling in the recruitment process.
    # </div>

    # # Question 2 : Statistical Tests

    # Pour répondre à la premiere question on utilise Chi2 Test
    hypothesis_test_evaluation.chi2_test(data, "sexe", "specialite")

    # <div class="alert alert-block alert-success">
    # Since the p-value is less than a significant level of 0.05, then we can reject the hypothesis that the two columns are independent and we can conclude that <b>the two features specialty and sex present a significant level of dependence</b>.
    # </div>
    cat_col = 'cheveux'
    cont_col = 'salaire'
    hypothesis_test_evaluation.assess_dependence_using_anova_test(data, cat_col, cont_col)

    # <div class="alert alert-block alert-success">
    # The null hypothesis of the ANOVA test is that the means of the continuous variable are equal in the different categories of the categorical variable. Since the P-value obtained by the ANOVA test is less than 0.05, we can reject the null hypothesis and conclude that the means are significantly different, indicating that there is <b>a significant dependence between the hair variable and the wage variable</b>.
    # </div>
    cat_col = "exp"
    cont_col = "note"

    hypothesis_test_evaluation.assess_dependence_using_anova_test(data, cat_col, cont_col)

    # <div class="alert alert-block alert-success">
    # The null hypothesis of the ANOVA test is that the means of the continuous variable are equal in the different categories of the categorical variable. Since the P-value obtained by the ANOVA test is greater than 0.05, we cannot reject the null hypothesis and conclude that the means are significantly different, which indicates that there is <b>no significant dependence between the experience feature and the score feature</b>.
    # </div>

    # # Question 3 : Model Fitting

    # ### 1. Filter unuseful data

    # <div class="alert alert-block alert-success">
    # As there is no significant dependency between each of these variables ['date', 'month', 'sex', 'availability', 'season', 'year'] and the variable we want to predict (hiring). So I chose to omit them in my proposal.
    # </div>

    to_drop = ['date', 'month', 'sexe', 'dispo', 'season', 'year']
    data.drop(columns=to_drop, inplace=True)

    # ### 2. Encode Categorical data

    # <div class="alert alert-block alert-success">
    # In order to use the categorical variables in our models, we need to transform them: There are two types of categorical data: <b>Nominal and Ordinal</b>. For nominal variables, we will transform them using one hot encoding and for ordinal variables, we will use the label Encoding.
    # </div>

    # <div class="alert alert-block alert-info">
    # <b>One hot encoding</b> is a way to represent categorical variables as numerical data. It involves creating a new column for each unique category in a categorical variable. Each row is then marked with a 1 in the column for the corresponding category and 0 in all other new columns.
    # </div>

    # <div class="alert alert-block alert-info">
    # <b>Label encoding</b> is a way to represent categorical variables as numerical data. It involves assigning a unique integer value to each category and then encoding the categories as integers. One potential issue with label encoding is that it can imply an ordinal relationship between the categories, where in reality there may not be one.
    # </div>

    one_hot_encoding_vars = ['cheveux', 'specialite']

    label_encoding_vars = ['diplome']

    features_encoders = FeaturesEncoders()

    data = features_encoders.one_hot_encoding(data, one_hot_encoding_vars)

    # We have an order between the different categories of the variable diploma, for this, we must make the encoding manually
    data = features_encoders.label_encoding_2(data, 'diplome', {"bac": 1, "licence": 2, 'master': 3, 'doctorat': 4})

    features = [col for col in data.columns if col != "embauche"]

    # ### 3. Model Choice

    # <div class="alert alert-block alert-info">
    # It is well known that tree-based algorithms : <b>Decision trees, random forests, and gradient boosting are all commonly used algorithms for classification tasks, and they are generally able to handle unbalanced data well</b>. These algorithms create a hierarchy of decision rules, which can be effective in identifying patterns in the data, even if the classes are unbalanced. In what follows, we will verify these assumptions by testing different algorithms with default configurations.
    # </div>

    model_fitter = ModelFitter(data)

    classifiers = [LogisticRegression(), SGDClassifier(), DecisionTreeClassifier(), \
                   RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

    for classifier in classifiers:
        model_fitter.fit(classifier)

    model_fitter.evaluate()

    # <div class="alert alert-block alert-success">
    # From these results, we notice that we have good results in term of accuracy, but bad for the rest, especially the F1-Score. Howver, as the accuracy is misleading in this case because the data are not balanced. Therefore, we used other metrics such as:
    #
    # - Precesion: the precision is the number of true positive predictions made by the model, divided by the total number of positive predictions made by the model. Precision is a good metric to use when the negative class is more prevalent, as it gives more weight to the correct identification of the minority class. In fact, in an unbalanced data set where the negative class is more prevalent, the classifier may be more likely to predict the negative class in order to achieve higher overall accuracy. However, this may lead to low accuracy, as there may be a high number of false positive predictions (negative instances that are incorrectly classified as positive). By using precision as a performance measure, we can ensure that the model is evaluated on its ability to correctly identify the positive class, rather than on its overall precision.
    #
    # - Recall: Recall is the number of true positive predictions made by the model, divided by the total number of true positive instances in the data. Recall is a good metric to use when the positive class is rarer, as it will give more weight to correctly identifying the minority class.
    #
    # - F1 Score: The F1 score is the harmonic mean of precision and recall, and is a good metric to use when you want to balance precision and recall.
    #
    # In the following, we will present some suggestions to improve it:
    # - Balancing the data using undersampling (oversampling using nc-SMOTE could be done also)
    # - Tuning the parameters of the best algorithm (in this case, based on the F1-score, the RFs are the best)
    # - Use AutoML in order to select the best algorithm with its parameters (will not be done here, since my OS does not support AutoML)
    # </div>

    #
    # ### 1. Balancing data using undersampling

    data_balancer = DataBalancer()

    balanced_data = data_balancer.under_sampling(data)

    model_fitter = ModelFitter(balanced_data)

    classifiers = [LogisticRegression(), SGDClassifier(), DecisionTreeClassifier(), \
                   RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

    for classifier in classifiers:
        model_fitter.fit(classifier)

    model_fitter.evaluate()

    # <div class="alert alert-block alert-success">
    # We see here that the performance of these classifiers has improved remarkably. Therefore, this confirms the fact that unbalanced data has a significant impact on the learning quality of an ML model. Nevertheless, we are still far from an acceptable prediction quality.
    # </div>

    # ### 2. Fine tuning Random Forests

    # <div class="alert alert-block alert-info">
    # There are several important parameters of a random forest model that can affect its performance:
    #
    # - <b>n_estimators</b>: This is the number of trees in the forest. A larger number of trees will typically lead to a more accurate model, but at the cost of increased computation time.
    #
    # - <b>max_depth</b>: This is the maximum depth of the trees in the forest. A larger maximum depth will result in more complex trees, which can lead to overfitting if the trees are allowed to grow too deep.
    #
    # - <b>min_samples_split</b>: This is the minimum number of samples required to split an internal node in the tree. A larger minimum sample size will result in simpler trees, which can reduce the risk of overfitting.
    #
    # - <b>min_samples_leaf</b>: This is the minimum number of samples required to be at a leaf node. A larger minimum sample size will result in simpler trees, which can reduce the risk of overfitting.
    #
    # - <b>max_features</b>: This is the maximum number of features considered when splitting a node in the tree. A larger number of features will result in more complex trees, which can lead to overfitting if the model is allowed to consider too many features.
    #
    # - <b>class_weight</b>: This is a weighting applied to the classes in the dataset. If the class distribution is imbalanced, adjusting the class weights can help the model to better handle the imbalanced data.
    #
    # - <b>random_state</b>: This is a seed value used to initialize the random number generator, which is used to randomly select features and samples when building the trees. Setting a fixed random state can be useful for reproducibility of results.
    # </div>

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=10)]

    # Number of features to consider at every split
    max_features = ['sqrt']

    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=10)]
    max_depth.append(None)

    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]

    class_weight = ['balanced', 'balanced_subsample']
    # bootstrap: Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
    bootstrap = [False]

    # <div class="alert alert-block alert-success">
    # To fine tune these parameters, we perform a grid search over the specified hyperparameter values using 5-fold cross-validation.
    # The GridSearchCV class will fit and evaluate the model for each combination of hyperparameters,
    # and the best combination will be chosen based on the model's performance on the validation data.
    #
    # For the sake of simplicity, we use RandomizedSearchCV which is similar to GridSearchCV, but instead of searching over a grid of hyperparameter values, it samples from a distribution of possible hyperparameter values. This can be more efficient than GridSearchCV when the hyperparameter search space is large, because it does not need to evaluate every possible combination of hyperparameters.
    # </div>

    # Define the model
    model = RandomForestClassifier(random_state=12)

    # Define the hyperparameter grid
    param_grid = {'n_estimators': n_estimators,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf,
                  'max_features': max_features,
                  # 'class_weight':class_weight,
                  'bootstrap': bootstrap}

    # Create a grid search object
    # grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

    randomized_rearch_rf = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=5,
                                              cv=3, verbose=2, random_state=12, n_jobs=-1)

    model_fitter = ModelFitter(data)

    model_fitter.fit(randomized_rearch_rf)

    model_fitter.evaluate()

    randomized_rearch_rf.best_params_

    model_fitter_balanced = ModelFitter(balanced_data)

    model_fitter_balanced.fit(randomized_rearch_rf)

    model_fitter_balanced.evaluate()

    # <div class="alert alert-block alert-success">
    # To conclude, using both of the data undersampling and parameters tuning have improved considerably the performances. Neverthless, the quality of predictions could be more enhanced if we use an oversampling approach like NC-SMOTE.
    # </div>
