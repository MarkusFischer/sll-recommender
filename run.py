import os

import numpy as np
from scipy import sparse
from sklearn import model_selection as ms

#todo Konstanten auslagern
import metrices.accuracy as accuracy
from knn.distance_metrics import pearson, cosine
from knn.knn import kNN
from models.bayes import NaiveBayes
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

classificator = NaiveBayes(X_train, [1,2,3,4,5],alpha=0.6)
classificator.fit()

y_hat = []
print(f"Lines: {X_test_raw.shape[0]}")

line_c = 1
for line in X_test_raw[:300,:].tolist():
    print(line_c)
    line_c += 1
    y_hat.append(classificator.predict(line[0], line[1],mode=0))

rmse_bayes = accuracy.rmse(X_test_raw[:300,2], np.array(y_hat)-1)
mae_bayes = accuracy.mae(X_test_raw[:300,2], np.array(y_hat)-1)
print(rmse_bayes)
print(mae_bayes)

#print("item based")
#y_hat = []
#for line in X_test_raw.tolist():
#    y_hat.append(classificator.predict(line[0], line[1],mode=1))

#rmse_bayes = accuracy.rmse(X_test_raw[:,2], np.array(y_hat)-1)
#mae_bayes = accuracy.mae(X_test_raw[:,2], np.array(y_hat)-1)
#print(rmse_bayes)
#print(mae_bayes)


#print("saving to file")
#Xq = np.genfromtxt(os.path.join("data", "qualifying_blanc.csv"), delimiter=",", dtype=np.int)
#bayes = []
#for line in Xq.tolist():
#    bayes.append(classificator.predict(line[0], line[1],mode=0))
#bayes = np.array(bayes)-1
#Xq_bayes = np.column_stack((Xq,bayes))
#np.savetxt("qualifying_bayes_first.csv", Xq_bayes.astype(np.int),
#           delimiter=",", newline="\n", encoding="utf-8")



#knn = kNN(X_train,3,pearson,cosine)

#y_hat_knn = []
#line_c = 1
#for line in X_test_raw[:100,:].tolist():
#    print(line_c)
#    line_c += 1
#    y_hat_knn.append(knn.classify(line[0], line[1],dir=0))

#rmse_knn = accuracy.rmse(X_test_raw[:100,2], np.array(y_hat_knn)-1)
#print(rmse_knn)

