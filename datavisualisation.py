import os

import numpy as np
import matplotlib.pyplot as plt

import os
import pickle

import numpy as np

from sklearn import model_selection as ms

from metrices import accuracy
from models.matrix_factorization import UMF

from utility.matrices import convert_sparse_coo_to_full_matrix

plt.xkcd()

X_raw = np.genfromtxt(os.path.join("data", "train.csv"), delimiter=",", dtype=np.int)
X_raw[:,2] = 1
X_raw = convert_sparse_coo_to_full_matrix(X_raw).toarray()
plt.figure()
plt.matshow(X_raw)
plt.savefig(os.path.join("img", "train_matrix.png"))

Xq = np.genfromtxt(os.path.join("data", "qualifying_blanc.csv"), delimiter=",", dtype=np.int)
q = np.full((Xq.shape[0],1),2)
X_q = convert_sparse_coo_to_full_matrix(np.column_stack((Xq, q))).toarray()
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

#rmse ~ rank
X_raw = np.genfromtxt(os.path.join("data", "train.csv"), delimiter=",", dtype=np.int)
X_train_raw, X_remaining_raw = ms.train_test_split(X_raw, test_size=0.1, random_state=42)
X_validate_raw, X_test_raw = ms.train_test_split(X_remaining_raw, test_size=0.1, random_state=42)

ranks = [1, 5, 10, 15, 20, 25, 30]
rmses = []
for rank in ranks:
    filename = "umf_rank_" + str(rank) + ".pyc"
    model = pickle.load(open(os.path.join("trained_models", filename), "rb"))
    rmse_train = accuracy.rmse(X_train_raw[:,2]-1,model.predict(X_train_raw[:,(0,1)])-1)
    rmse_validate = accuracy.rmse(X_validate_raw[:, 2], model.predict(X_validate_raw[:, (0, 1)]) - 1)
    absolute_error = model.learn_insights[-1][1]
    rmses.append((rank, rmse_train, rmse_validate, absolute_error))
plt.figure()
rmses = np.array(rmses)
plt.plot(rmses[:,0],rmses[:,2], color="blue", marker="o")
plt.title("RMSE on validation set vs. matrix rank")
plt.ylabel("RMSE")
plt.xlabel("rank")
plt.savefig(os.path.join("img","rank_rmse.png"))
plt.figure()
plt.plot(rmses[:,0],rmses[:,3], color="blue", marker="o")
plt.savefig(os.path.join("img","rank_abs_error.png"))