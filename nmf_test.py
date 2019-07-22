import os
import pickle

import numpy as np

from sklearn import model_selection as ms

from metrices import accuracy
from models.matrix_factorization import UMF, NMF

X_raw = np.genfromtxt(os.path.join("data", "train.csv"), delimiter=",", dtype=np.int)
X_train_raw, X_remaining_raw = ms.train_test_split(X_raw, test_size=0.1, random_state=42)
X_validate_raw, X_test_raw = ms.train_test_split(X_remaining_raw, test_size=0.1, random_state=42)


print("data preprocessing")
X_train_raw[:,2] += 1

print(f"Building estimator with rank 20")
factorizer = NMF(X_train_raw, rank=20, epsilon=1e-4, max_run=25_000, verbose=True)
factorizer.fit()

rmse = accuracy.rmse(X_test_raw[:,2], factorizer.predict(X_test_raw[:,(0,1)])-1)
print(f"RMSE for rank 20: {rmse}")

filename = "nmf_25_000.pyc"
pickle.dump(factorizer, open(os.path.join("trained_models",filename), "wb"))