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
        y_hat = 1 / (1 + np.exp(-decision_boundary))
        return y_hat

    def __loss_func(self, W, X, y):
        l = 0
        for idx, row in enumerate(X):
            y_hat = self.__hypothesis(W, row)
            l += (-y[idx]*np.log(y_hat) - (1 - y[idx])*np.log(1 - y_hat))
        return l

    def fit(self, X, y, lr, epochs):  # or gradient descent
        for row in X:
            row.insert(0, 1)
        init_w = [1] * len(X[0])

        for _ in range(epochs):
            for row_idx, row in enumerate(X):
                for idx, w in enumerate(init_w):
                    if idx == 0:
                        init_w[idx] = w - lr * (self.__hypothesis(init_w, row) - y[row_idx])
                    else:
                        init_w[idx] = w - lr * (self.__hypothesis(init_w, row) - y[row_idx]) * row[idx]
            # print(f'Epoch {i}, loss: ', self.__loss_func(init_w, X, y))
        self.weights = init_w

    def predict(self, data_point):
        prediction = self.__hypothesis(self.weights, data_point)
        return prediction

    def evaluation(self, X_val, y_val):
        predict_true = list()
        for idx, row in enumerate(X_val):
            prediction = self.predict(row)
            if round(prediction) == y_val[idx]:
                predict_true.append(prediction)
        accuracy = len(predict_true) / len(X_val)
        return accuracy
 