import numbers

import numpy as np

from knn.distance_metrics import SimiliarityMatrix

class kNN:
    def __init__(self,
                 data,
                 user_similarity,
                 item_similarity,
                 row_mean=None,
                 k=3):
        self.data = data

        self.data_bin = np.zeros_like(data)
        self.data_bin[data != 0] = 1

        if user_similarity.shape[0] != user_similarity.shape[1] or user_similarity.shape[0] != data.shape[0]:
            raise Exception("User similarity matrix is not a square matrix or dimension mismatch!")
        self.user_similarity = user_similarity
        if item_similarity.shape[0] != item_similarity.shape[1] or item_similarity.shape[0] != data.shape[1]:
            raise Exception("Item similarity matrix is not a square matrix or dimension mismatch!")
        self.item_similarity = item_similarity
        if row_mean is not None:
            self.row_mean = row_mean
        else:
            self.row_mean = np.zeros((data.shape[0],1))
        self.data_mean_free = data - row_mean
        if not isinstance(k, numbers.Number) or k <= 0:
            raise Exception(f"K value {k} is to small or not a number!")#todo check if not int
        self.k = k


    def classify(self,coords,axis=None):
        if axis == 0:
            sim = []
            ratings = []
            for (u,j) in coords[:,(0,1)]:#todo sieh nach ob die item j überhaupt bewertet haben
                nearest = np.argpartition(self.user_similarity[u,:], -(self.k+1))[-(self.k+1):]
                nearest = nearest[nearest != u]
                sim.append(self.user_similarity[u,nearest])
                ratings.append(self.data_mean_free[nearest,j])
            sim = np.array(sim)
            ratings = np.array(ratings)
            return self.row_mean[coords[:,0]].T + (np.sum(sim*ratings,axis=1))/(np.sum(np.abs(sim),axis=1))
        pass


