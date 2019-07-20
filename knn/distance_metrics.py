import pickle

import numpy as np

from utility.matrices import make_rows_mean_free


class SimiliarityMatrix:
    def __init__(self,
                 data_matrix,
                 axis=0,
                 mean="row",
                 method="pearson",
                 verbose=False):
        self.data = data_matrix
        if axis != 0 and axis != 1:
            raise Exception("Wrong value for axis")
        self.axis = axis
        self.similarity = np.eye(data_matrix.shape[axis])
        self.mean = np.zeros((data_matrix.shape[axis],1))
        self.verbose = verbose
        if method.lower() != "pearson" and method.lower() != "cosine":
            raise Exception("Uncommon method!")
        self.method = method.lower()
        if mean.lower() != "row" and mean.lower() != "col":
            raise Exception("Only rows or columns can made mean free.")
        self.mean_method = mean

    def save(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self.similarity, file)
        with open("mean_" + filename, "wb") as file:
            pickle.dump(self.mean, file)


    def fit(self):
        data_bin = np.zeros_like(self.data)
        data_bin[self.data != 0] = 1
        if self.mean_method == "row":
            row_sum = np.sum(self.data, axis=1)
            entry_count = np.sum(data_bin, axis=1)
            mean = (row_sum / (entry_count+10e-9)).reshape(-1,1)
            self.mean = mean
            mean_free_data = self.data - mean
        else:
            row_sum = np.sum(self.data, axis=0)
            entry_count = np.sum(data_bin, axis=0)
            mean = (row_sum / (entry_count+10e-9)).reshape(-1, 1)
            self.mean = mean
            mean_free_data = (self.data.T - mean).T
        for u in range(0,self.data.shape[self.axis]):
            if self.verbose:
                print(f"Running {u} of {self.data.shape[self.axis]}")
            for v in range(u + 1, self.data.shape[self.axis]):
                if self.method == "pearson":
                    if self.axis == 0:
                        items_u = mean_free_data[u,:][data_bin[u,:] == data_bin[v,:]]
                        items_v = mean_free_data[v, :][data_bin[u,:] == data_bin[v,:]]
                    else:
                        items_u = mean_free_data[:,u][data_bin[:,u] == data_bin[:,v]]
                        items_v = mean_free_data[:,v][data_bin[:,u] == data_bin[:,v]]
                else:
                    if self.axis == 0:
                        items_u = self.data[u,:][data_bin[u,:] == data_bin[v,:]]
                        items_v = self.data[v, :][data_bin[u,:] == data_bin[v,:]]
                    else:
                        items_u = self.data[:,u][data_bin[:,u] == data_bin[:,v]]
                        items_v = self.data[:,v][data_bin[:,u] == data_bin[:,v]]
                self.similarity[u, v] = np.sum(items_u * items_v) / (10e-9 +
                            np.sqrt(np.sum(items_u ** 2)) * np.sqrt(np.sum(items_v ** 2)))
        lower_indices = np.tril_indices(self.data.shape[self.axis],-1)
        self.similarity[lower_indices] = self.similarity.T[lower_indices]


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
    psm = SimiliarityMatrix(example,axis=1)
    psm.fit()