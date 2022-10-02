import numpy as np


class LogisticRegressionFC:
    def __init__(self):
        self.final_weight = []

    def h(self, W, X_c):
        '''
        Parameters
        ----------
        X_c: stands for a row in dataset
        '''
        decision_boundary = 0
        for idx, feature_value in enumerate(X_c):
            decision_boundary += W[idx] * feature_value
        y_hat = 1 / (1 + np.exp(-decision_boundary))
        print(y_hat)
        return y_hat

    def loss_func(self, W, X, y):
        l = 0
        for idx, row in enumerate(X):
            y_hat = self.h(W, row)
            l += (-y[idx]*np.log(y_hat) - (1-y[idx])*np.log(1-y_hat))
        return l

    def fit(self, X, y, lr, epochs):  # or gradient descent
        for row in X:
            row.insert(0, 1)
        init_w = [0.01] * len(X[0])

        for _ in range(epochs):
            for row_idx, row in enumerate(X):
                for idx, w in enumerate(init_w):
                    if idx == 0:
                        init_w[idx] = w - lr * (y[row_idx] - self.h(init_w, row))
                    else:
                        init_w[idx] = w - lr * (y[row_idx] - self.h(init_w, row)) * row[idx]
            print(self.loss_func(init_w, X, y))
        self.final_weight = init_w

    def predict(self, data_point):
        prediction = 0
        for i, feature in enumerate(data_point):
            prediction += self.final_weight[i] * feature
        return prediction

    # def evaluate(self, X_val, y_val):
    #     rmse = 0
    #     for i, row in enumerate(X_val):
    #         y_val_predicted = self.predict(row)
    #         rmse += pow((y_val[i] - y_val_predicted), 2)
    #     return math.sqrt(rmse/len(X_val))


if __name__ == '__main__':
    import numpy as np
    logit_reg_fc = LogisticRegressionFC()
    X = np.random.randint(1, 5, size=(500, 5)).tolist()
    y = np.random.randint(0, 2, size=(500, )).tolist()
    logit_reg_fc.fit(X, y, 0.00001, 20)
