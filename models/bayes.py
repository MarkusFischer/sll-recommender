import operator

import numpy as np


class NaiveBayes:
    def __init__(self, train_matrix, levels, alpha=0):
        self.matrix = train_matrix.toarray() #TODO check if it is already an numpy array...
        self.alpha = alpha
        self.levels = levels
        self.fitted = False
        pass

    def fit(self):
        self.matrix_decompositions = {"total":self.matrix}
        self.matrix_row_entry_count = {"total":np.zeros(self.matrix.shape[0])}
        self.matrix_column_entry_count = {"total":np.zeros(self.matrix.shape[1])}
        for level in self.levels:
            temp = np.full_like(self.matrix, level)
            temp[(self.matrix - level).nonzero()] = 0
            self.matrix_decompositions[level] = temp
            row_count = np.sum(temp,axis=1)/level
            self.matrix_row_entry_count[level] = row_count
            self.matrix_row_entry_count["total"] = self.matrix_row_entry_count["total"] + row_count
            col_count = np.sum(temp, axis=0) / level
            self.matrix_column_entry_count[level] = col_count
            self.matrix_column_entry_count["total"] = self.matrix_column_entry_count["total"] + col_count
        self.fitted = True

    def __probability_rui_is_vs(self, i, v):
        if not self.fitted:
            raise Exception("model not fitted")
        return (self.matrix_column_entry_count[v][i] + self.alpha)/(self.matrix_column_entry_count["total"][i] + len(self.levels)*self.alpha)

    def __probability_ruk_cond_ruj_is_vs(self, u, k, j, v):
        if not self.fitted:
            raise Exception("model not fitted")
        ruk = self.matrix[u,k]
        return (self.matrix_column_entry_count[ruk][k] + self.alpha)/(self.matrix_column_entry_count[v][j] + len(self.levels)*self.alpha)

    def predict(self, u, i, mode=None): #TODO vectorized
        if mode is None:
            pass
        elif mode == 0: #row (user) based
            #items rated by user u
            Ru = self.matrix[u,:]
            Iu = Ru.nonzero()[0].tolist() #todo Vector?
            predicts = {}
            for level in self.levels:
                naive = 1
                for k in Iu:
                    naive *= self.__probability_ruk_cond_ruj_is_vs(u, k, i, level)
                predicts[level] = naive*self.__probability_rui_is_vs(i, level)
            return max(predicts.items(), key=operator.itemgetter(1))[0]
        elif mode == 1: #column (item) based
            pass