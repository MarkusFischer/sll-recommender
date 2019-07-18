from scipy import sparse
import numpy as np


def convert_sparse_coo_to_full_matrix(matrix):
    return sparse.coo_matrix((matrix[:,2], (matrix[:,0], matrix[:,1]))).tocsr()

def make_rows_mean_free(matrix):
    row_sums = matrix.sum(axis=1)
    print(row_sums.shape)
    nnz = matrix.getnnz(axis=1).reshape((-1,1))
    nnz[nnz[:]==0] = 1
    mean = row_sums/nnz.reshape((-1,1))
    return (matrix - mean, mean)
