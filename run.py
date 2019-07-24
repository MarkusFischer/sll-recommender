import os
import pickle

import numpy as np
from sklearn import model_selection as ms

#todo Konstanten auslagern
import metrices.accuracy as accuracy
from models.matrix_factorization import UMF

X_raw = np.genfromtxt(os.path.join("data", "train.csv"), delimiter=",", dtype=np.int)
X_train_raw, X_test_raw = ms.train_test_split(X_raw, test_size=0.1)

print("data preprocessing")
X_train_raw[:,2] += 1

final_model = UMF(X_train = X_train_raw,
                  rank=19,
                  regularization=4,
                  eta="lsearch",
                  epsilon=1e-2,
                  max_run=5000,
                  bias=True,
                  verbose=True)
final_model.fit()
pickle.dump(final_model, open(os.path.join("trained_models", "final_model.pyc"), "wb"))

print(f"RMSE for final model: {accuracy.rmse(X_test_raw[:,2], final_model.predict(X_test_raw[:,(0,1)])-1)}")
print("saving to file")
Xq = np.genfromtxt(os.path.join("data", "qualifying_blanc.csv"), delimiter=",", dtype=np.int)
bayes = final_model.predict(Xq)-1
bayes[np.nonzero(bayes < 0)] = 0
bayes[np.nonzero(bayes > 4)] = 4
Xq_final = np.column_stack((Xq,bayes))
np.savetxt("qualifying_final.csv", Xq_final.astype(np.int),
           delimiter=",", newline="\n", encoding="utf-8")



#knn = kNN(X_train,3,pearson,cosine)

#y_hat_knn = []
#line_c = 1
#for line in X_test_raw[:100,:].tolist():
#    print(line_c)
#    line_c += 1
#    y_hat_knn.append(knn.classify(line[0], line[1],dir=0))

#rmse_knn = accuracy.rmse(X_test_raw[:100,2], np.array(y_hat_knn)-1)
#print(rmse_knn)

