import os

import numpy as np
import matplotlib.pyplot as plt

from utility.matrices import convert_coo_to_sparse

X_raw = np.genfromtxt(os.path.join("data", "train.csv"), delimiter=",", dtype=np.int)
X_raw[:,2] = 1
X_raw = convert_coo_to_sparse(X_raw).toarray()
plt.figure()
plt.matshow(X_raw)
plt.savefig(os.path.join("img", "train_matrix.png"))

Xq = np.genfromtxt(os.path.join("data", "qualifying_blanc.csv"), delimiter=",", dtype=np.int)
q = np.full((Xq.shape[0],1),2)
X_q = convert_coo_to_sparse(np.column_stack((Xq, q))).toarray()
plt.figure()
plt.matshow(X_raw+X_q)
plt.savefig(os.path.join("img", "train_and_qualify.png"))

#item count
plt.figure()
item_count = np.sum(X_raw, axis=0)
plt.bar(np.arange(0,item_count.shape[0]), item_count)
plt.savefig(os.path.join("img", "item_count.png"))

plt.figure()
user_count = np.sum(X_raw, axis=1)
plt.bar(np.arange(0,user_count.shape[0]), user_count)
plt.savefig(os.path.join("img", "user_count.png"))