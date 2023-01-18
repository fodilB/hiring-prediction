__author__ = 'BENALI Fodil'
__email__ = 'fodel.benali@gmail.com'
__copyright__ = 'Copyright (c) 2023, Technical Test'

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ModelFitter:

    def __init__(self, data, depedent_variable="embauche", test_size=0.1, random_state=42):

        data = data.reset_index(drop=True)

        features = [col for col in data.columns if col != depedent_variable]

        X = data[features]

        y = data[depedent_variable]

        self.predictions = {}

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size,
                                                                                random_state=random_state)

    def fit(self, model):

        model_name = type(model).__name__

        model.fit(self.X_train, self.y_train)

        self.predictions[model_name] = model.predict(self.X_test)

    def evaluate(self):

        comparison_table = [['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score']]

        for model in self.predictions.keys():
            pred = self.predictions[model]
            model_scores = [model, round(accuracy_score(self.y_test, pred), 4), \
                            round(precision_score(self.y_test, pred), 4), \
                            round(recall_score(self.y_test, pred), 4), \
                            round(f1_score(self.y_test, pred), 4)]
            comparison_table.append(model_scores)

        for line in comparison_table:
            print(f'{line[0]:<25}', end='')
            for ele in line[1:]:
                print(f'{ele:<15}', end='')
            print()
