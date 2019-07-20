import os

import numpy as np
from sklearn import model_selection as ms

import metrices.accuracy as accuracy
from utility.matrices import convert_sparse_coo_to_full_matrix

X_raw = np.genfromtxt(os.path.join("data", "train.csv"), delimiter=",", dtype=np.int)
X_train_raw, X_test_raw = ms.train_test_split(X_raw, test_size=0.1, random_state=42)

print("Data preprocesing")
X_train_raw[:,2] += 1

X_train = convert_sparse_coo_to_full_matrix(X_train_raw).toarray()

X_train_bin = np.zeros_like(X_train)
X_train_bin[X_train != 0] = 1


print("row means")
row_sum = np.sum(X_train, axis=1)
entry_count = np.sum(X_train_bin, axis=1)
entry_count[entry_count == 0] = 1
row_mean = row_sum/entry_count
y_hat_row = row_mean[X_test_raw[:,0]]
print(f"RMSE {accuracy.rmse(X_test_raw[:,2], y_hat_row)}")

print("col means")
col_sum = np.sum(X_train, axis=0)
entry_count = np.sum(X_train_bin, axis=0)
entry_count[entry_count == 0] = 1
col_mean = col_sum/entry_count
y_hat_row = col_mean[X_test_raw[:,1]]
print(f"RMSE {accuracy.rmse(X_test_raw[:,2], y_hat_row)}")