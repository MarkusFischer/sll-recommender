import pickle

import numpy as np

from utility.matrices import make_rows_mean_free


class PearsonSimiliarityMatrix:
    def __init__(self,
                 data_matrix,
                 axis=0,
                 verbose=False):
        self.data = data_matrix
        if axis != 0 and axis != 1:
            raise Exception("Wrong value for axis")
        self.axis = axis
        self.similarity = np.eye(data_matrix.shape[axis])
        self.verbose = verbose

    def save(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self.similarity, file)

    def fit(self):
        row_sum = np.sum(self.data, axis=1-self.axis)
        data_bin = np.zeros_like(self.data)
        data_bin[self.data != 0] = 1
        entry_count = np.sum(data_bin, axis=1-self.axis)
        mean = (row_sum / entry_count).reshape(-1,1)
        if self.axis == 0:
            mean_free_data = self.data - mean
        else:
            mean_free_data = self.data.T - mean
        for u in range(0,self.data.shape[self.axis]):
            print(f"Running {u} of {self.data.shape[self.axis]}")
            for v in range(u + 1, self.data.shape[self.axis]):
                if self.axis == 0:
                    items_u = mean_free_data[u,:][data_bin[u,:] == data_bin[v,:]]
                    items_v = mean_free_data[v, :][data_bin[u,:] == data_bin[v,:]]
                    self.similarity[u,v] = np.sum(items_u*items_v)/(np.sqrt(np.sum(items_u**2))*np.sqrt(np.sum(items_v**2)))
                else:
                    items_u = mean_free_data[:,u][data_bin[:,u] == data_bin[:,v]]
                    items_v = mean_free_data[:,v][data_bin[:,u] == data_bin[:,v]]
                    self.similarity[u, v] = np.sum(items_u * items_v) / (
                                np.sqrt(np.sum(items_u ** 2)) * np.sqrt(np.sum(items_v ** 2)))
        lower_indices = np.tril_indices(self.data.shape[self.axis],-1)
        self.similarity[lower_indices] = self.similarity.T[lower_indices]


def pearson(u,v,matrix,mean):
    rows = matrix.shape[0]
    if (u > rows or v > rows):
        return 0
    nonzeros = np.transpose(matrix.nonzero())
    Iu = set(nonzeros[nonzeros[:,0]==u][:,1])
    Iv = set(nonzeros[nonzeros[:,0]==v][:,1])
    I_common = Iu & Iv
    ru = np.empty((0,0))
    rv = np.empty((0,0))
    for k in I_common: #Vektorisieren????
        ru = np.append(ru, matrix[u,k]-mean[u])
        rv = np.append(rv, matrix[v,k]-mean[v])
    return np.sum(ru*rv)/(np.sqrt(np.sum(ru*ru))*np.sqrt(np.sum(rv*rv)))

def cosine(u,v,matrix, mean):
    rows = matrix.shape[0]
    if (u > rows or v > rows):
        return 0
    nonzeros = np.transpose(matrix.nonzero())
    Iu = set(nonzeros[nonzeros[:, 0] == u][:, 1])
    Iv = set(nonzeros[nonzeros[:, 0] == v][:, 1])
    I_common = Iu & Iv
    ru = np.empty((0, 0))
    rv = np.empty((0, 0))
    for k in I_common: #Vektorisieren????
        ru = np.append(ru, matrix[u,k])
        rv = np.append(rv, matrix[v,k])
    return np.sum(ru*rv)/(np.sqrt(np.sum(ru*ru))*np.sqrt(np.sum(rv*rv)))


if __name__ == "__main__":
    example = np.array([[7, 6, 7, 4, 5, 4],
                        [6, 7, 0, 4, 3, 4],
                        [0, 3, 3, 1, 1, 0],
                        [1, 2, 2, 3, 3, 4],
                        [1, 0, 1, 2, 3, 3]])
    row_sum = np.sum(example, axis=1)
    example_bin = np.zeros_like(example)
    example_bin[example != 0] = 1
    entry_count = np.sum(example_bin, axis=1)
    mean = row_sum/entry_count
    psm = PearsonSimiliarityMatrix(example)
    psm.fit()