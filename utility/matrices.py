from scipy import sparse


def convert_sparse_coo_to_full_matrix(matrix):
    return sparse.coo_matrix((matrix[:,2], (matrix[:,0], matrix[:,1]))).tocsr()
