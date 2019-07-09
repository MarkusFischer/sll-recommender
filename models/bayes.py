import operator
import warnings

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

    def __probability_rui_is_vs(self, i, v, transpose=False):
        if not self.fitted:
            raise Exception("model not fitted")
        if transpose:
            return (self.matrix_row_entry_count[v][i] + self.alpha) / (
                    self.matrix_row_entry_count["total"][i] + len(self.levels) * self.alpha)
        else:
            return (self.matrix_column_entry_count[v][i] + self.alpha) / (
                    self.matrix_column_entry_count["total"][i] + len(self.levels)*self.alpha)

    def __probability_ruk_cond_ruj_is_vs(self, u, k, j, v,transpose=False):
        if not self.fitted:
            raise Exception("model not fitted")
        if transpose:
            rvj = self.matrix[k, j]
        else:
            ruk = self.matrix[u, k]
        if transpose:
            return (self.matrix_row_entry_count[rvj][k] + self.alpha) / (
                    self.matrix_row_entry_count[v][j] + len(self.levels) * self.alpha)
        else:
            #Anzahl der Nutzer für das k-te Item Rating ruk vergeben: self.matrix_column_entry_count[ruk][k]
            #Anzahl der Nutzer für das k-te Item Rating v vergeben: self.matrix_column_entry_count[v][k]
            return (len(np.intersect1d(self.matrix_decompositions[ruk][:,k].nonzero()[0],
                                       self.matrix_decompositions[v][:,j].nonzero()[0])) + self.alpha) / (
                    self.matrix_column_entry_count[v][j] +  len(self.levels) * self.alpha )
            #return (self.matrix_column_entry_count[v][j] + self.alpha) / (
            #        self.matrix_column_entry_count[ruk][k] + len(self.levels)*self.alpha)

    def predict(self, u, i, mode=None): #TODO vectorized
        if mode is None:
            pass
        elif mode == 0: #row (user) based
            #items rated by user u
            Ru = self.matrix[u,:]
            Iu = Ru.nonzero()[0].tolist() #todo Vector?
            predicts = {}
            for level in self.levels:
                naive = self.__probability_rui_is_vs(i, level)
                for k in Iu:
                    temp = self.__probability_ruk_cond_ruj_is_vs(u, k, i, level)
                    if (temp > 1):
                        warnings.warn(f"probability for {u},{i},{k} = {temp} > 1", Warning)
                    naive *= temp
                predicts[level] = naive
            return max(predicts.items(), key=operator.itemgetter(1))[0]
        elif mode == 1: #column (item) based
            Ci = self.matrix[:,i]
            Ui = Ci.nonzero()[0].tolist()  # todo Vector?
            predicts = {}
            for level in self.levels:
                naive = 1
                for v in Ui:
                    naive *= self.__probability_ruk_cond_ruj_is_vs(v, i, i, level, transpose=True)
                predicts[level] = naive * self.__probability_rui_is_vs(u, level, transpose=True)
            return max(predicts.items(), key=operator.itemgetter(1))[0]