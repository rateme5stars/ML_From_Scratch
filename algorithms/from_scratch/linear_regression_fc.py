class LinearRegressionFC:
    def __init__(self):
        self.final_weight = []

    def h(self, W, X_c):
        '''
        Parameters
        ----------
        X_c: stands for a row in dataset
        '''
        y_hat = 0
        for idx, feature_value in enumerate(X_c):
            y_hat += W[idx] * feature_value
        return y_hat

    def loss_func(self, W, X, y):
        l = 0
        for idx, row in enumerate(X):
            y_hat = self.h(W, row)
            l += 0.5 * pow((y_hat - y[idx]), 2) / len(X)
        return l

    def train(self, X, y, lr, epochs):  # or gradient descent
        for row in X:
            row.insert(0, 1)
        init_w = [1] * len(X[0])

        for _ in range(epochs):
            for row_idx, row in enumerate(X):
                for idx, w in enumerate(init_w):
                    if idx == 0:
                        init_w[idx] = w - (self.h(init_w, row) - y[row_idx]) * lr
                    else:
                        init_w[idx] = w - lr * (self.h(init_w, row) - y[row_idx]) * row[idx]
            print(self.loss_func(init_w, X, y))
        self.final_weight = init_w

    def predict(self, data_point):
        data_point.insert(1, 0)
        prediction = 0
        for i, feature in enumerate(data_point):
            prediction += self.final_weight[i] * feature
        return prediction
