import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from mllearn.preprocessing.normalize import Normalize
from mllearn.algorithms.linear_regression.linear_regression_fc import LinearRegressionFC


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
    # Linear Regression from scratch test
    df = pd.read_csv('mllearn/dataset/insurance.csv')
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
    print('--- Linear Regression from scratch test ---')
    lrfc.fit(X, y, 0.003, 10)
    print(f'RMSE:', lrfc.evaluate(X_val, y_val))

    # Compare with Linear Regression from sklearn
    from sklearn import linear_model
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error

    print('\n--- Linear Regression from sklearn test ---')

    dataset2 = data_processing(df)
    minmax_scaler = MinMaxScaler()
    dataset2 = minmax_scaler.fit_transform(dataset2)
    X_train, X_eval, y_train, y_eval = dataset2[:1001, :-1], dataset2[1001:, :-1], dataset2[:1001, -1], dataset2[1001:, -1]
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)
    y_eval_predict = lr.predict(X_eval)
    rmse = np.sqrt(mean_squared_error(y_eval, y_eval_predict))
    print('RMSE:', rmse)
