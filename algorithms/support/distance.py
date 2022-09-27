import math

class Distance:
    # Distance between sample vs specific data point in dataset
    @staticmethod
    def euclidean(x1=None, x2=None):
        distance = 0
        for i in range(len(x1)):
            distance += (x1[i] - x2[i])**2
        return math.sqrt(distance)

    @staticmethod
    def manhattan(x1=None, x2=None):
        distance = 0
        for i in range(len(x1)):
            distance += abs(x1[i] - x2[i])
        return distance
