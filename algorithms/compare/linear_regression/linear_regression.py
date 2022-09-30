import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from algorithms.from_scratch.linear_regression_fc import LinearRegressionFC
from algorithms.support.normalize import Normalize


def data_processing(dataset):
    ohe = OneHotEncoder()
    ohe_feature = ohe.fit_transform(dataset[['region']]).toarray()
    feature_name = ohe.categories_
    ohe_df = pd.DataFrame(ohe_feature, columns=feature_name)
    dataset = pd.concat([dataset, ohe_df], axis=1)
    dataset['sex'] = np.where(dataset['sex'] == 'male', 1, 0)
    dataset['smoker'] = np.where(dataset['smoker'] == 'yes', 1, 0)
    dataset.drop(['region'], axis=1, inplace=True)

    charges_col = dataset.pop('charges')
    dataset.insert(dataset.shape[1], 'charges', charges_col)

    return dataset


if __name__ == '__main__':
    df = pd.read_csv('algorithms/dataset/insurance.csv')
    dataset = data_processing(df).values.tolist()

    normalized_dataset = Normalize(dataset=dataset)
    minmax_scaled_dataset = normalized_dataset.minmax()

    y = []
    for row in minmax_scaled_dataset:
        y.append(row[-1])
        row.pop()
    X = minmax_scaled_dataset
    X_train, X_val = X[:1001], X[1001:]
    y_train, y_val = y[:1001], y[1001:]
    lrfc = LinearRegressionFC()
    lrfc.fit(X, y, 0.003, 20)
    print(f'RMES of FC:', lrfc.evaluate(X_val, y_val))
