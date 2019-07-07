import os

import numpy as np
from scipy import sparse
from sklearn import model_selection as ms

#todo Konstanten auslagern
import metrices.accuracy as accuracy
from knn.distance_metrics import pearson, cosine
from knn.knn import kNN
from utility.matrices import convert_coo_to_sparse, make_rows_mean_free

X_raw = np.genfromtxt(os.path.join("data", "train.csv"), delimiter=",", dtype=np.int)
print(f"maximum rating: {np.amax(X_raw[:,2])}")
print(f"minimum rating: {np.amin(X_raw[:,2])}")
X_train_raw, X_test_raw = ms.train_test_split(X_raw, test_size=0.1, random_state=42)
X_train = convert_coo_to_sparse(X_train_raw)
X_test = convert_coo_to_sparse(X_test_raw)

#simple test mse/rmse/mae with given data should be zero
mse = accuracy.mse(X_test_raw[:,2],X_test_raw[:,2])
rmse = accuracy.rmse(X_test_raw[:,2],X_test_raw[:,2])
mae = accuracy.mae(X_test_raw[:,2],X_test_raw[:,2])

#test with examples
example = np.array([[7,6,7,4,5,4],
                    [6,7,0,4,3,4],
                    [0,3,3,1,1,0],
                    [1,2,2,3,3,4],
                    [1,0,1,2,3,3]])

example_sparse = sparse.coo_matrix(example)
(matrix, mean) = make_rows_mean_free(X_train)

classificator = kNN(X_train,3,pearson,cosine)

y_hat = np.empty((0,0))

print(f"Lines: {X_test_raw.shape[0]}")
line_c = 0
def calc(line):
    global line_c
    line_c += 1
    print(line_c)
    np.append(y_hat, classificator.classify(line[0], line[1],dir=0))

np.apply_along_axis(calc, axis=1, arr=X_test_raw)

rmse = accuracy.rmse(X_test_raw[:,2], y_hat)
print(rmse)