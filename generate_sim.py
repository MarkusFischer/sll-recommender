import os

import numpy as np
from sklearn import model_selection as ms

from knn.distance_metrics import SimiliarityMatrix
from utility.matrices import convert_sparse_coo_to_full_matrix

X_train_raw = np.genfromtxt(os.path.join("data", "train.csv"), delimiter=",", dtype=np.int)
#X_train_raw, X_remaining_raw = ms.train_test_split(X_raw, test_size=0.1, random_state=42)
#X_validate_raw, X_test_raw = ms.train_test_split(X_remaining_raw, test_size=0.1, random_state=42)


print("data preprocessing")
X_train_raw[:,2] += 1


data = convert_sparse_coo_to_full_matrix(X_train_raw).toarray()


print("calculate pearson similarity matrix for user; dimensional reduction")
pearson = SimiliarityMatrix(data,axis=0,verbose=True, reduce_dimension=False)
pearson.fit()
pearson.save("user_pearson_sim.pyc")

#print("calculate cosine similarity matrix for user")
#cosine = SimiliarityMatrix(data, axis=0,method="cosine",verbose=True)
#cosine.fit()
#cosine.save("user_cosine_sim.pyc")


print("calculate pearson similarity matrix for item; dimensional reduction")
pearson = SimiliarityMatrix(data,axis=1,verbose=True,reduce_dimension=False)
pearson.fit()
pearson.save("item_pearson_sim.pyc")

#print("calculate cosine similarity matrix for user")
#cosine = SimiliarityMatrix(data, axis=1,method="cosine",verbose=True)
#cosine.fit()
#cosine.save("item_cosine_sim.pyc")