import os
import pickle

import numpy as np

from sklearn import model_selection as ms

from metrices import accuracy
from models.matrix_factorization import UMF

X_raw = np.genfromtxt(os.path.join("data", "train.csv"), delimiter=",", dtype=np.int)
X_train_raw, X_remaining_raw = ms.train_test_split(X_raw, test_size=0.1, random_state=42)
X_validate_raw, X_test_raw = ms.train_test_split(X_remaining_raw, test_size=0.1, random_state=42)


print("data preprocessing")
X_train_raw[:,2] += 1

ranks = [1, 5, 10, 15, 20, 25, 30, 35]

for rank in ranks:
    print(f"Building estimator with rank {rank}")
    factorizer = UMF(X_train_raw, rank=rank, eta=0.0005, regularization=0, epsilon=1e-3, max_run=7500, verbose=True,
                     bias=False)
    factorizer.fit()

    rmse = accuracy.rmse(X_test_raw[:,2], factorizer.predict(X_test_raw[:,(0,1)])-1)
    print(f"RMSE for rank {rank}: {rmse}")

    filename = "umf_rank_" + str(rank) + "_no_bias.pyc"
    pickle.dump(factorizer, open(os.path.join("trained_models",filename), "wb"))

    print(f"Building estimator with rank wrt. bias {rank}")
    factorizer_bias = UMF(X_train_raw, rank=rank, eta=0.0005, regularization=0, epsilon=1e-3, max_run=7500, verbose=True,
                     bias=True)
    factorizer_bias.fit()

    rmse = accuracy.rmse(X_test_raw[:, 2], factorizer_bias.predict(X_test_raw[:, (0, 1)]) - 1)
    print(f"RMSE for rank {rank}: {rmse}")

    filename = "umf_rank_" + str(rank) + "_bias.pyc"
    pickle.dump(factorizer, open(os.path.join("trained_models", filename), "wb"))
