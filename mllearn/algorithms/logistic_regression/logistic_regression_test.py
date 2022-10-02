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
    dataset = dataset.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
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
    print('--- Logistic Regression from scratch test ---')
    logit_reg.fit(X_train, y_train, 0.00018, 30)
    accuracy = logit_reg.evaluation(X_val, y_val)
    print('Accuracy:', accuracy)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score

    print('\n--- Logistic Regression from sklearn test ---')
    dataset2 = data_processing(df)
    minmax_scaler = MinMaxScaler()
    dataset2 = minmax_scaler.fit_transform(dataset2)

    X_train, X_eval, y_train, y_eval = dataset2[:701, 1:], dataset2[701:, 1:], dataset2[:701, 0], dataset2[701:, 0]

    logit_reg = LogisticRegression()
    logit_reg.fit(X_train, y_train)

    y_eval_predict = logit_reg.predict(X_eval)
    accuracy = accuracy_score(y_eval, y_eval_predict)

    print('Accuracy:', accuracy)
