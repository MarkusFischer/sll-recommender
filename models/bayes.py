import operator

import numpy as np


class NaiveBayes:
    def __init__(self, train_matrix, levels, alpha=0):
        self.matrix = train_matrix
        self.alpha = alpha
        self.levels = levels
        pass

    def __probability_rui_is_vs(self, i, v):
        rating_count = {} #no map but array?
        rows = self.matrix.shape[0]
        for level in self.levels:
            temp = self.matrix[self.matrix[:,i].nonzero()[0],i].toarray()-v
            rating_count[level] = rows - np.count_nonzero(temp) #TODO axis
        return (rating_count[v]+self.alpha)/(sum(rating_count.values()) + len(self.levels)*self.alpha)

    def __probability_ruk_cond_ruj_is_vs(self, u, k, j, v):
        ruk = self.matrix[u,k]
        rows = self.matrix.shape[0]
        users_rated_item_k_to_ruk = rows - np.count_nonzero(self.matrix[self.matrix[:,k].nonzero()[0],k].toarray() - ruk) #TODO axis
        users_rated_item_j_to_v = rows - np.count_nonzero(self.matrix[self.matrix[:,j].nonzero()[0],j].toarray() - v)
        return (users_rated_item_k_to_ruk + self.alpha)/(users_rated_item_j_to_v + len(self.levels)*self.alpha)

    def predict(self, u, i, mode=None): #TODO vectorized
        if mode is None:
            pass
        elif mode == 0: #row (user) based
            #items rated by user u
            Ru = self.matrix[u,:]
            Iu = Ru.nonzero()[1].tolist() #todo Vector?
            predicts = {}
            for level in self.levels:
                naive = 1
                for k in Iu:
                    naive *= self.__probability_ruk_cond_ruj_is_vs(u, k, i, level)
                predicts[level] = naive*self.__probability_rui_is_vs(i, level)
            return max(predicts.items(), key=operator.itemgetter(1))[0]
        elif mode == 1: #column (item) based
            pass