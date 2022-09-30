from random import randrange
import pandas as pd
import numpy as np
from sklearn import datasets


# from algorithms.from_scratch.knn_fc import KnnFC

iris = datasets.load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                  columns=iris['feature_names'] + ['target'])
# X, y = iris.data, iris.target
dataset = df.values.tolist()


def k_fold_split(dataset, n_folds):
    dataset_split = list()
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset))
            fold.append(dataset.pop(index))
        dataset_split.append(fold)
    return dataset_split


def evaluate(dataset, n_folds, algorithms=None):
    folds = k_fold_split(dataset, n_folds)
    score = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
    return train_set


if __name__ == '__main__':
    print(evaluate(dataset, 10))
