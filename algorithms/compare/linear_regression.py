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
    lrfc = LinearRegressionFC()

    lrfc.train(X, y, 0.00015, 5)
