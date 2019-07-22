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

#lambdas = [0.2, 0.4, 0.6, 0.8, 1, 1.5, 2]
lambdas = [1.25, 1.75, 2.5, 3, 3.5, 4]

for lambd in lambdas:
    print(f"Building estimator with lambda {lambd}")
    factorizer = UMF(X_train_raw, rank=20, eta=0.0005, regularization=lambd, epsilon=1e-3, max_run=5000, verbose=True,
                     bias=True)
    factorizer.fit()

    rmse = accuracy.rmse(X_test_raw[:,2], factorizer.predict(X_test_raw[:,(0,1)])-1)
    print(f"RMSE for lambda {lambd}: {rmse}")

    filename = "umf_lambda_" + str(lambd) + ".pyc"
    pickle.dump(factorizer, open(os.path.join("trained_models",filename), "wb"))
