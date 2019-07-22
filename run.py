import os
import pickle

import numpy as np
from scipy import sparse
from sklearn import model_selection as ms
from sklearn.decomposition import nmf as nmf

#todo Konstanten auslagern
import metrices.accuracy as accuracy
from models.matrix_factorization import UMF, NMF

X_raw = np.genfromtxt(os.path.join("data", "train.csv"), delimiter=",", dtype=np.int)
print(f"maximum rating: {np.amax(X_raw[:,2])}")
print(f"minimum rating: {np.amin(X_raw[:,2])}")
X_train_raw, X_test_raw = ms.train_test_split(X_raw, test_size=0.1, random_state=42)

print("data preprocessing")
X_train_raw[:,2] += 1

print("fitting with sgd")
#classificator_sgd = UMF(X_train_raw, rank=45, random_state=42, eta=5e-5, regularization=0.7, epsilon=1, max_run=200, method="gd")
#classificator_sgd.fit(verbosity=1)


print("fitting with NMF of sklearn")
#predictor = nmf.NMF(n_components=150,
#                    solver='mu',
#                    verbose=True,
#                    max_iter=1000,
#                    alpha=0.5,
#                    tol=1e-10)
#res = predictor.fit_transform(convert_sparse_coo_to_full_matrix(X_train_raw))
#result = predictor.inverse_transform(res)
#print(result.shape)
#predictions = result[X_test_raw[:,0],X_test_raw[:,1]]


print("fitting with NMF")
#recommender = NMF(X_train_raw, rank=30, random_state=42, regularization=None, epsilon=1000, stability=10e-9, max_run=1300)
#recommender.fit(verbosity=1)

print("fitting with classic gd")
#classificator = UMF(X_train_raw, rank=15, random_state=42, eta=0.0005, regularization=0, epsilon=1e-5, max_run=200, verbose=True)
#classificator.fit(verbosity=1)

#pickle.dump(classificator, open("umf_test.pyc", "wb"))



#rmse_gd = accuracy.rmse(X_test_raw[:,2], classificator.predict(X_test_raw[:,(0,1)])-1)
#mae_gd = accuracy.mae(X_test_raw[:,2], classificator.predict(X_test_raw[:,(0,1)])-1)
#print(f"RMSE gd: {rmse_gd}")
#print(f"MAE gd: {mae_gd}")

#rmse_sgd = accuracy.rmse(X_test_raw[:,2], recommender.predict(X_test_raw[:,(0,1)])-1)
#rmse_sgd = accuracy.rmse(X_test_raw[:,2], classificator_sgd.predict(X_test_raw[:,(0,1)])-1)
#mae_sgd = accuracy.mae(X_test_raw[:,2], recommender.predict(X_test_raw[:,(0,1)])-1)
#mae_sgd = accuracy.mae(X_test_raw[:,2], classificator_sgd.predict(X_test_raw[:,(0,1)])-1)
#print(f"RMSE sgd: {rmse_sgd}")
#print(f"MAE sgd: {mae_sgd}")

#rmse_sklearn = accuracy.rmse(X_test_raw[:,2],predictions-1)
#mae_sklearn = accuracy.mae(X_test_raw[:,2],predictions-1)
#print(f"RMSE sklearn: {rmse_sklearn}")
#print(f"MAE sklearn: {mae_sklearn}")



#print("item based")
#y_hat = []
#for line in X_test_raw.tolist():
#    y_hat.append(classificator.predict(line[0], line[1],mode=1))

#rmse_bayes = accuracy.rmse(X_test_raw[:,2], np.array(y_hat)-1)
#mae_bayes = accuracy.mae(X_test_raw[:,2], np.array(y_hat)-1)
#print(rmse_bayes)
#print(mae_bayes)



classificator = pickle.load(open("trained_models/umf_lambda_4.pyc", "rb"))
print("saving to file")
Xq = np.genfromtxt(os.path.join("data", "qualifying_blanc.csv"), delimiter=",", dtype=np.int)
bayes = classificator.predict(Xq)-1
bayes[np.nonzero(bayes < 0)] = 0
bayes[np.nonzero(bayes > 4)] = 4
#for line in Xq.tolist():
#    bayes.append(classificator.predict(line[0], line[1]))
#bayes = np.array(bayes)-1
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

