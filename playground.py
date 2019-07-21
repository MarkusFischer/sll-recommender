import os
import pickle

import numpy as np

from knn.distance_metrics import SimiliarityMatrix
from knn.knn import kNN
from metrices import accuracy

example = np.array([[7, 6, 7, 4, 5, 4],
                        [6, 7, 0, 4, 3, 4],
                        [0, 3, 3, 1, 1, 0],
                        [1, 2, 2, 3, 3, 4],
                        [1, 0, 1, 2, 3, 3]])
user_sim = SimiliarityMatrix(example, axis=0)
user_sim.fit()
item_sim = SimiliarityMatrix(example, axis=1)
item_sim.fit()
classificator = kNN(example, user_sim.similarity, item_sim.similarity, user_sim.mean,k=2)
example_res = classificator.classify(np.array([[2,0],[2,5]]),axis=1)
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
X_train_raw[:,2] += 1

data = convert_sparse_coo_to_full_matrix(X_train_raw).toarray()

useful = kNN(data, user, item, mean, k=77)
#useful.classify(np.array([[2121,375]]),axis=0)
y_hat = useful.classify(X_remaining_raw[:,(0,1)],axis=0)
rmse_knn = accuracy.rmse(X_remaining_raw[:,2], y_hat-1)
y_hat2 = useful.classify(X_remaining_raw[:,(0,1)],axis=1)
rmse_knn2 = accuracy.rmse(X_remaining_raw[:,2], y_hat2-1)
y_hat3 = useful.classify(X_remaining_raw[:,(0,1)])
rmse_knn3 = accuracy.rmse(X_remaining_raw[:,2], y_hat3-1)