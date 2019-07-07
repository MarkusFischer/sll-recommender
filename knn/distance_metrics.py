import numpy as np


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