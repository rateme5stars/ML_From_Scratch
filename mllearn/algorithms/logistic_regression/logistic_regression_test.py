from locale import normalize
import pandas as pd
import numpy as np
from logistic_regression_fc import LogisticRegressionFC
from mllearn.preprocessing.normalize import Normalize


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


def data_processing(dataset):
    dataset['Age'] = dataset[['Age', 'Pclass']].apply(impute_age, axis=1)
    dataset['Sex'] = np.where(dataset['Sex'] == 'male', 1, 0)
    dataset.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    return dataset


if __name__ == '__main__':
    df = pd.read_csv('mllearn/dataset/titanic.csv')
    dataset = data_processing(df).values.tolist()

    normalize = Normalize(dataset=dataset)
    X = normalize.minmax()

    y = []
    for row in X:
        y.append(row[0])
        row.pop(0)

    X_train, X_val = X[:701], X[701:]
    y_train, y_val = y[:701], y[701:]
    logit_reg = LogisticRegressionFC()
    logit_reg.fit(X_train, y_train, 0.0001, 1)
