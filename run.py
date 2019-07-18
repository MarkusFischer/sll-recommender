import os

import numpy as np
from scipy import sparse
from sklearn import model_selection as ms

#todo Konstanten auslagern
import metrices.accuracy as accuracy
from knn.distance_metrics import pearson, cosine
from knn.knn import kNN
from models.bayes import NaiveBayes
from models.matrix_factorization import UMF, NMF
from utility.matrices import convert_sparse_coo_to_full_matrix, make_rows_mean_free

X_raw = np.genfromtxt(os.path.join("data", "train.csv"), delimiter=",", dtype=np.int)
print(f"maximum rating: {np.amax(X_raw[:,2])}")
print(f"minimum rating: {np.amin(X_raw[:,2])}")
X_train_raw, X_test_raw = ms.train_test_split(X_raw, test_size=0.1, random_state=42)

print("data preprocessing")
X_train_raw[:,2] += 1

print("fitting with sgd")
#classificator_sgd = UMF(X_train_raw, rank=5, random_state=2, eta=5e-6, regularization=None, epsilon=1, max_run=1000, method="sgd")
#classificator_sgd.fit(verbosity=1)

print("fitting with NMF")
recommender = NMF(X_train_raw, rank=5, random_state=42, regularization=None, epsilon=1000, max_run=200)
recommender.fit(verbosity=1)

#print("fitting with classic gd")
#classificator = UMF(X_train_raw, rank=5 , random_state=2, eta=0.000005, regularization=None, epsilon=0.5, max_run=500)
#classificator.fit(verbosity=1)



#rmse_gd = accuracy.rmse(X_test_raw[:,2], classificator.predict(X_test_raw[:,0:1])-1)
#mae_gd = accuracy.mae(X_test_raw[:,2], classificator.predict(X_test_raw[:,0:1])-1)
#print(f"RMSE gd: {rmse_gd}")
#print(f"MAE gd: {mae_gd}")

rmse_sgd = accuracy.rmse(X_test_raw[:,2], recommender.predict(X_test_raw[:,(0,1)])-1)
mae_sgd = accuracy.mae(X_test_raw[:,2], recommender.predict(X_test_raw[:,(0,1)])-1)
print(f"RMSE sgd: {rmse_sgd}")
print(f"MAE sgd: {mae_sgd}")


#print("item based")
#y_hat = []
#for line in X_test_raw.tolist():
#    y_hat.append(classificator.predict(line[0], line[1],mode=1))

#rmse_bayes = accuracy.rmse(X_test_raw[:,2], np.array(y_hat)-1)
#mae_bayes = accuracy.mae(X_test_raw[:,2], np.array(y_hat)-1)
#print(rmse_bayes)
#print(mae_bayes)


print("saving to file")
Xq = np.genfromtxt(os.path.join("data", "qualifying_blanc.csv"), delimiter=",", dtype=np.int)
bayes = recommender.predict(Xq)
#for line in Xq.tolist():
#    bayes.append(classificator.predict(line[0], line[1]))
bayes = np.array(bayes)-1
Xq_bayes = np.column_stack((Xq,bayes))
np.savetxt("qualifying_bayes_first.csv", Xq_bayes.astype(np.int),
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

