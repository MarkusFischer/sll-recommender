import os

import numpy as np
import matplotlib.pyplot as plt

from utility.matrices import convert_sparse_coo_to_full_matrix

A = np.array([[1, 2, 3, 0],
              [0, 0, 2, 0],
              [2, 1, 0, 0],
              [0, 2, 2, 0]])

values = np.unique(A)[1:] #0 nicht vergeben


#fuer jeden Wert aus values:
#hier beispielsweise 1
A1 = np.full_like(A,1)
A1[(A-1).nonzero()]=0

A2 = np.full_like(A,2)
A2[(A-2).nonzero()]=0

A3 = np.full_like(A,3)
A3[(A-3).nonzero()]=0

#Anzahl der belegten Reihen mit Wert i
row_count = np.sum(A2,axis=1)/2

X_raw = np.genfromtxt(os.path.join("data", "train.csv"), delimiter=",", dtype=np.int)
X_raw[:,2] += 1
Xq = np.genfromtxt(os.path.join("data", "qualifying_blanc.csv"), delimiter=",", dtype=np.int)
q = np.full((Xq.shape[0],1),7)
X_raw_mat = convert_sparse_coo_to_full_matrix(X_raw)
X_q_mat = convert_sparse_coo_to_full_matrix(np.column_stack((Xq, q)))
res = X_raw_mat.toarray() + X_q_mat.toarray()
plt.matshow(res)
plt.colorbar()
plt.show()