import os
import pickle

import numpy as np

from knn.distance_metrics import SimiliarityMatrix
from knn.knn import kNN
from metrices import accuracy
from sklearn import model_selection as ms

from models.matrix_factorization import UMF
from utility.matrices import convert_sparse_coo_to_full_matrix

X_raw = np.genfromtxt(os.path.join("data", "train.csv"), delimiter=",", dtype=np.int)
X_train_raw, X_remaining_raw = ms.train_test_split(X_raw, test_size=0.1, random_state=42)
X_validate_raw, X_test_raw = ms.train_test_split(X_remaining_raw, test_size=0.1, random_state=42)


print("data preprocessing")
X_train_raw[:,2] += 1


factorizer = UMF(X_train_raw, rank=15, eta="lsearch", regularization=0, epsilon=1e-3, max_run=500, verbose=True,
                     bias=True)
factorizer.fit()

def __loss_function(uvt, user_count, ratings):
    U = uvt[:user_count,:]
    V = uvt[user_count:,:]
    predicted = np.matmul(U, V.T)
    E = ratings - predicted
    E[np.nonzero(ratings == 0)] = 0
    return 0.5 * np.sum(E*E)

def __gradient_loss_function(uvt, user_count, ratings):
    U = uvt[:user_count, :]
    V = uvt[user_count:, :]
    predicted = np.matmul(U, V.T)
    E = ratings - predicted
    E[np.nonzero(ratings == 0)] = 0
    gradU = -np.matmul(E, V)
    gradV = -np.matmul(E.T, U)
    return np.row_stack((gradU, gradV))