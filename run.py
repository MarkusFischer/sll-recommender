import os

import numpy as np
import scipy
from sklearn import model_selection as ms

#todo Konstanten auslagern
import metrices.accuracy as accuracy
from utility.matrices import convert_coo_to_sparse

X_raw = np.genfromtxt(os.path.join("data", "train.csv"), delimiter=",", dtype=np.int)
X_train_raw, X_test_raw = ms.train_test_split(X_raw, test_size=0.1, random_state=42)
X_train = convert_coo_to_sparse(X_train_raw)
X_test = convert_coo_to_sparse(X_test_raw)

#simple test mse/rmse/mae with given data should be zero
mse = accuracy.mse(X_test_raw[:,2],X_test_raw[:,2])
rmse = accuracy.rmse(X_test_raw[:,2],X_test_raw[:,2])
mae = accuracy.mae(X_test_raw[:,2],X_test_raw[:,2])