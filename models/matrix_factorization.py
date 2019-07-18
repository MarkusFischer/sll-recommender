import random

import numpy as np

from metrices.accuracy import rmse, mae
from utility.matrices import convert_coo_to_sparse


class UMF:
    def __init__(self,
                    X_train,
                    rank=10,
                    random_state=random.randint(1, 101),
                    regularization=0.5,
                    eta=0.2,
                    epsilon=1,
                    max_run=1_000,
                    method="GD"
                ):
        if X_train.shape[1] != 3:
            raise Exception("wrong shape")
        self.train = X_train
        self.train_full = convert_coo_to_sparse(X_train).toarray()
        self.m = self.train_full.shape[0]
        self.n = self.train_full.shape[1]
        self.rank = rank
        self.random_state=random_state
        self.eta = eta
        self.epsilon = epsilon
        self.max_run = max_run
        self.regularization = regularization
        np.random.seed(self.random_state)
        #todo regularization and method implementation

    def fit(self, verbose=True):
        if verbose:
            print("initialize U and V with random values")
        self.U = np.random.rand(self.m, self.rank)
        self.V = np.random.rand(self.n, self.rank)
        y = self.train[:, 2]
        E_old = self.train_full
        for i in range(1, self.max_run):
            R = np.matmul(self.U, self.V.T)
            E = (self.train_full - R)
            E[np.nonzero(self.train_full == 0)] = 0
            if verbose:
                #print(f"Cylce: {i} of {self.max_run} error: {rmse(y,R[self.train[:,0],self.train[:,1]])}")
                #print(f"Cylce: {i} of {self.max_run} error: {np.sum(np.abs(E*E))}")
                print(f"Cycle: {i} of {self.max_run} error(step_size): {0.5*np.abs(np.sum(E*E - E_old*E_old))}")
                #print(f"step size: {}")
                print(f"max: {np.max(R)}")
                print(f"min: {np.min(R)}")
            if 0.5*np.abs(np.sum(E*E - E_old*E_old)) <= self.epsilon:
            #if np.all(np.abs(E*E) <= self.epsilon):
            #if rmse(y,R[self.train[:,0],self.train[:,1]]) <= self.epsilon:
                #Konvergenz erreicht
                if verbose:
                    print("Convergency reached!")
                    print(f"error: {rmse(y,np.max(np.abs(E*E)))}")
                    #print(f"error: {rmse(y,R[self.train[:,0],self.train[:,1]])}")

                break
            U = self.U
            V = self.V
            if self.regularization is not None:
                self.U = self.U - self.eta*self.regularization*self.U + self.eta * np.matmul(E, self.V)
                self.V = self.V - self.eta*self.regularization*self.V + self.eta * np.matmul(E.T, U)
            else:
                self.U = U + self.eta*np.matmul(E,V)
                self.V = V + self.eta*np.matmul(E.T,U)
            E_old = E
        self.result = np.matmul(self.U, self.V.T)

    def predict(self, u, i):
        return self.result[u,i]