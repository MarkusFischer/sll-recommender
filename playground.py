import os
import pickle

import numpy as np

from knn.distance_metrics import SimiliarityMatrix
from knn.knn import kNN
from metrices import accuracy

example = np.array([[5, 6, 7, 4, 3, 0],
                        [4, 0, 3, 0, 5, 4],
                        [0, 3, 4, 1, 1, 0],
                        [7, 4, 3, 6, 0, 4],
                        [1, 0, 3, 2, 2, 5]])
user_sim = SimiliarityMatrix(example, axis=0)
user_sim.fit()
item_sim = SimiliarityMatrix(example, axis=1)
item_sim.fit()
classificator = kNN(example, user_sim.similarity, item_sim.similarity, user_sim.mean,k=2)
example_res = classificator.classify(np.array([[0,5],[1,1],[1,3],[2,0],[2,5],[3,4],[4,1]]),axis=0)
#example_res = classificator.classify(np.array([[2,0]]), axis=0)

user = pickle.load(open("user_pearson_sim.pyc", "rb"))
mean = pickle.load(open("mean_user_pearson_sim.pyc", "rb"))
item = pickle.load(open("item_pearson_sim.pyc", "rb"))
from sklearn import model_selection as ms

from utility.matrices import convert_sparse_coo_to_full_matrix

X_raw = np.genfromtxt(os.path.join("data", "train.csv"), delimiter=",", dtype=np.int)
X_train_raw, X_remaining_raw = ms.train_test_split(X_raw, test_size=0.1, random_state=42)
X_validate_raw, X_test_raw = ms.train_test_split(X_remaining_raw, test_size=0.1, random_state=42)


print("data preprocessing")
X_raw[:,2] += 1

data = convert_sparse_coo_to_full_matrix(X_raw).toarray()

useful = kNN(data, user, item, mean, k=15)
useful.classify(np.array([[2121,375]]),axis=0)
y_hat = useful.classify(X_remaining_raw[:,(0,1)],axis=0)
rmse_knn = accuracy.rmse(X_remaining_raw[:,2], y_hat-1)
mae_knn = accuracy.mae(X_remaining_raw[:,2], y_hat-1)
y_hat2 = useful.classify(X_remaining_raw[:,(0,1)],axis=1)
rmse_knn2 = accuracy.rmse(X_remaining_raw[:,2], y_hat2-1)
y_hat3 = useful.classify(X_remaining_raw[:,(0,1)])
rmse_knn3 = accuracy.rmse(X_remaining_raw[:,2], y_hat3-1)