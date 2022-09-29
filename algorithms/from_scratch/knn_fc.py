from algorithms.support.distance import Distance

class KnnFC:
    def __init__(self, dataset, trainset, distance_method):
        self.dataset = dataset
        self.trainset = trainset
        self.distance_method = distance_method
        
        if distance_method == 'euclidean':
            self.distance_method = Distance().euclidean
        elif distance_method == 'manhattan':
            self.distance_method= Distance().manhattan
        else:
            print('Error: distance_method must be "euclidean" or "manhattan"')

    def get_sorted_neighbors(self, sample):
        neighbors = list()
        for i, data_point in enumerate(self.dataset):
            distance = self.distance_method(sample, data_point)
            neighbors.append([distance, self.trainset[i]])
        neighbors.sort(key=lambda tup: tup[0])
        return neighbors

    def predict(self, k_neighbors, sample):
        sorted_neighbors = self.get_sorted_neighbors(sample=sample)
        neighbors = sorted_neighbors[:k_neighbors]
        outputs = [n[-1] for n in neighbors]
        prediction = max(set(outputs), key=outputs.count)
        return prediction
                                                                                                           