import numpy as np

class LogisticRegressionFC:
    def __init__(self):
        self.weights = []

    def __hypothesis(self, W, X_c):
        '''
        Parameters
        ----------
        X_c: stands for a row in dataset
        '''
        decision_boundary = 0
        for idx, feature_value in enumerate(X_c):
            decision_boundary += (W[idx] * feature_value)
            print(idx)
        print('start')
        y_hat = 1 / (1 + np.exp(-decision_boundary))
        print(y_hat)
        return y_hat

    def loss_func(self, W, X, y):
        l = 0
        for idx, row in enumerate(X):
            y_hat = self.__hypothesis(W, row)
            l += (-y[idx]*np.log(y_hat) - (1 - y[idx])*np.log(1 - y_hat))
        return l

    def fit(self, X, y, lr, epochs):  # or gradient descent
        for row in X:
            row.insert(0, 1)
        init_w = [0] * len(X[0])

        for _ in range(epochs):
            for row_idx, row in enumerate(X):
                for idx, w in enumerate(init_w):
                    if idx == 0:
                        init_w[idx] = w - lr * (y[row_idx] - self.__hypothesis(init_w, row))
                    else:
                        init_w[idx] = w - lr * (y[row_idx] - self.__hypothesis(init_w, row)) * row[idx]
            self.loss_func(init_w, X, y)
        self.weights = init_w

    def predict(self, data_point):
        prediction = 0
        for i, feature in enumerate(data_point):
            prediction += self.weights[i] * feature
        return prediction
 