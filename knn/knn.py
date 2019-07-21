import numbers

import numpy as np

from knn.distance_metrics import SimiliarityMatrix

class kNN:
    def __init__(self,
                 data,
                 user_similarity,
                 item_similarity,
                 row_mean=None,
                 k=3,
                 alpha = 0.5):
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
        self.data_mean_free[data==0]=0
        if not isinstance(k, numbers.Number) or k <= 0:
            raise Exception(f"K value {k} is to small or not a number!")#todo check if not int
        self.k = k

        if not isinstance(alpha, numbers.Number) or alpha < 0 or alpha > 1:
            raise Exception("Alpha is not a numeric value in [0,1]")
        self.alpha = alpha


    def classify(self,coords,axis=None):
        if axis is None:
            user_based = self.classify(coords=coords,axis=0)
            item_based = self.classify(coords=coords,axis=1)
            return self.alpha * user_based + (1-self.alpha)*item_based
        elif axis == 0:
            sim = []
            ratings = []
            for (u,j) in coords[:,(0,1)]:
                nearest = np.argpartition(self.user_similarity[u,(self.data_bin[:,j]==1)], -(self.k))[-(self.k):]
                user_sim = self.user_similarity[u,self.data_bin[:,j]==1][nearest]
                user_sim.resize(self.k)
                sim.append(user_sim)
                data = self.data_mean_free[self.data_bin[:,j]==1,j][nearest]
                data.resize(self.k)
                ratings.append(data)
            sim = np.array(sim)
            ratings = np.array(ratings)
            num = np.sum(sim*ratings,axis=1)
            denom = np.sum(np.abs(sim),axis=1)
            result = np.divide(num,denom+10e-9)
            return np.add(self.row_mean[coords[:,0]].reshape(1,-1), result.reshape(1,-1))
        elif axis == 1:
            sim = []
            ratings = []
            for (u, j) in coords[:, (0, 1)]:
                try:
                    nearest = np.argpartition(self.item_similarity[j, (self.data_bin[u,:] == 1)], -self.k)[-self.k:]
                    item_sim = self.item_similarity[j, self.data_bin[u,:] == 1][nearest]
                    data = self.data[u, self.data_bin[u,:] == 1][nearest]
                except:
                    item_sim = self.item_similarity[j, self.data_bin[u,:] == 1]
                    data = self.data[u, self.data_bin[u,:] == 1]
                item_sim.resize(self.k)
                data.resize(self.k)
                sim.append(item_sim)
                ratings.append(data)
            sim = np.array(sim)
            ratings = np.array(ratings)
            num = np.sum(sim * ratings, axis=1)
            denom = np.sum(np.abs(sim), axis=1)
            result = np.divide(num, denom + 10e-9)
            return result.reshape(1, -1)
            pass


