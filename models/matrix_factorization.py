import random

import numpy as np

from metrices.accuracy import rmse, mae
from utility.matrices import convert_sparse_coo_to_full_matrix


class UMF:
    def __init__(self,          #TODO check each parameter for correct type
                    X_train,
                    rank=10,
                    random_state=random.randint(1, 101),
                    regularization=None,
                    eta=0.2,
                    epsilon=1,
                    max_run=1_000,
                    method="GD"
                ):
        if X_train.shape[1] != 3:
            raise Exception("wrong shape")
        self.train = X_train
        self.train_full = convert_sparse_coo_to_full_matrix(X_train).toarray()
        self.m = self.train_full.shape[0]
        self.n = self.train_full.shape[1]
        self.rank = rank
        self.random_state=random_state
        self.eta = eta
        self.epsilon = epsilon
        self.max_run = max_run
        self.regularization = regularization
        np.random.seed(self.random_state)
        self.method = method.lower()
        self.matrix = None

    def fit(self, verbosity=0, validation_set=None):
        if verbosity == 2 and validation_set is None:
            raise Exception("For usage of verbosity level 2 an validation set is required")
        if verbosity >= 1:
            print("initialize U and V with random values")
        U = np.random.rand(self.m, self.rank)
        V = np.random.rand(self.n, self.rank)

        if self.method == "gd":
            E_old = np.zeros((self.m,self.n))
            for i in range(1, self.max_run):
                R_predicted = np.matmul(U, V.T)
                E = (self.train_full - R_predicted)
                E[np.nonzero(self.train_full == 0)] = 0
                if verbosity >= 1:
                    print(f"Cycle: {i} of {self.max_run} error(frobenius): {0.5* np.abs(np.sum(E*E))} error(step_size): {0.5 * np.abs(np.sum(E * E - E_old * E_old))}")

                if 0.5 * np.abs(np.sum(E * E - E_old * E_old)) <= self.epsilon:
                    if verbosity >= 1:
                        print("Convergency reached!")
                    break

                U_old = U
                V_old = V
                if self.regularization is not None:
                    U = U - self.eta * self.regularization * U_old + self.eta * np.matmul(E, V_old)
                    V = V - self.eta * self.regularization * V_old + self.eta * np.matmul(E.T, U_old)
                else:
                    U = U_old + self.eta * np.matmul(E, V_old)
                    V = V_old + self.eta * np.matmul(E.T, U_old)
                E_old = E

        elif self.method == "sgd":
            E_old = np.zeros((self.m, self.n))
            for i in range(1, self.max_run):
                R_predicted = np.matmul(U, V.T)
                E = self.train_full - R_predicted
                E[np.nonzero(self.train_full == 0)] = 0

                if verbosity >= 1:
                    print(f"Cycle: {i} of {self.max_run} error(frobenius): {0.5* np.abs(np.sum(E*E))} error(stepsize): {0.5 * np.abs(np.sum(E * E - E_old * E_old))}")

#                if 0.5 * np.abs(np.sum(E * E - E_old * E_old)) <= self.epsilon:
                if 0.5*np.sum(E*E) <= self.epsilon:
                    if verbosity >= 1:
                        print("Convergency reached!")
                    break

                training_shuffeld = self.train
                np.random.shuffle(training_shuffeld)
                #for sample in range(0, training_shuffeld.shape[0]): #TODO regularization
                i = training_shuffeld[1, 0]
                j = training_shuffeld[1, 1]
                U_old = U
                V_old = V
                U[i,:] = U_old[i,:] + self.eta * E[i,j] * V_old[j,:]
                V[j, :] = V_old[j, :] + self.eta * E[i, j] * U_old[i,:]
                E_old = E
        else:
            raise Exception(f"{self.method} is not an valid learning algorithm. Currently only gradient descent and stochastic gradient descent are supported")
        self.matrix = np.matmul(U, V.T)

    def predict(self, coords):
        if not isinstance(coords, np.ndarray):
            raise Exception("Predict parameter is not of type np.ndarray or has wrong column count")
        return self.matrix[coords[:,0],coords[:,1]]

class NMF:
    def __init__(self,          #TODO check each parameter for correct type
                    X_train,
                    rank=10,
                    random_state=random.randint(1, 101),
                    regularization=None,
                    epsilon=1,
                    eta = 10**(-9),
                    max_run=1_000,
                ):
        if X_train.shape[1] != 3:
            raise Exception("wrong shape")
        self.train = X_train
        self.train_full = convert_sparse_coo_to_full_matrix(X_train).toarray()
        self.m = self.train_full.shape[0]
        self.n = self.train_full.shape[1]
        self.rank = rank
        self.random_state=random_state
        self.epsilon = epsilon
        self.max_run = max_run
        self.regularization = regularization
        np.random.seed(self.random_state)
        self.matrix = None
        self.eta = eta

    def fit(self, verbosity=0, validation_set = None):
        if verbosity == 2 and validation_set is None:
            raise Exception("For usage of verbosity level 2 an validation set is required")
        if verbosity >= 1:
            print("initialize U and V with random values")
        U = np.random.rand(self.m, self.rank)
        V = np.random.rand(self.n, self.rank)
        for i in range(0, self.max_run):
            R_predicted = np.matmul(U, V.T)
            E = self.train_full - R_predicted
            error = 0.5* np.sum(E*E)
            if verbosity >= 1:
                print(f"Cycle {i} of {self.max_run}; error: {error}")
            if error <= self.epsilon:
                break
            U_old = U
            V_old = V
            U_delta_numerator = np.multiply(U_old, np.matmul(self.train_full,V_old))
            U_delta_denumerator = np.matmul(U_old, np.matmul(V.T, V)) + self.eta
            U = np.divide(U_delta_numerator, U_delta_denumerator)

            V_delta_numerator = np.multiply(V_old, np.matmul(self.train_full.T, U_old))
            V_delta_denumerator = np.matmul(V_old, np.matmul(U.T, U)) + self.eta
            V = np.divide(V_delta_numerator, V_delta_denumerator)
        self.matrix = np.matmul(U,V.T)

    def predict(self, coords):
        if not isinstance(coords, np.ndarray):
            raise Exception("Predict parameter is not of type np.ndarray or has wrong column count")
        return self.matrix[coords[:,0],coords[:,1]]