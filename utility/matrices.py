import scipy


def convert_coo_to_sparse(matrix):
    return scipy.sparse.coo_matrix((matrix[:,2], (matrix[:,0], matrix[:,1])))