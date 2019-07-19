import numbers
import random

import numpy as np

from metrices.accuracy import rmse, mae
from utility.matrices import convert_sparse_coo_to_full_matrix


class UMF:
    def __init__(self,
                    X_train,
                    rank=10,
                    random_state=None,
                    regularization=0,
                    eta=0.2,
                    epsilon=1,
                    max_run=1_000,
                    method="gd",
                    convergence_check="step",
                    verbose=False
                ):
        if X_train.shape[1] != 3:
            raise Exception("Training set matrix has wrong shape!")
        self.train = X_train
        self.train_full = convert_sparse_coo_to_full_matrix(X_train).toarray()
        self.m = self.train_full.shape[0]
        self.n = self.train_full.shape[1]


        if type(rank) != int or rank <= 0:
            raise Exception("Rank must be an positiv integer value!")
        self.rank = rank
        if random_state is not None:
            np.random.seed(random_state)

        if not isinstance(eta, numbers.Number) and eta.lower() != "lsearch":
            raise Exception("Eta must be an numeric value or \"lsearch\" for searching with line_search for perfect eta!")
        self.eta = eta

        if not isinstance(epsilon, numbers.Number):
            raise Exception("Epsilon must be an numeric value!")
        self.epsilon = epsilon
        if type(max_run) != int or rank <= 0:
            raise Exception("Maximum run count must be an positiv integer value!")
        self.max_run = max_run
        if not isinstance(regularization, numbers.Number) or regularization < 0:
            raise Exception("The regularization parameter must be an non negative integer value!")
        self.regularization = regularization

        if method.lower() != "gd" and method.lower() != "sgd":
            raise Exception(f"{method} is not an valid learning algorithm. Currently only gradient descent and stochastic gradient descent are supported!")
        self.method = method.lower()

        if convergence_check.lower() != "step" and convergence_check.lower() != "value":
            raise Exception(f"{convergence_check} is not an supported convergence criteria. Currently only step size and absolut value are supported!")
        self.convergence_check = convergence_check.lower()

        self.matrix = np.zeros(self.train_full.shape)
        self.U = np.zeros((self.m, self.rank))
        self.V = np.zeros((self.n, self.rank))

        self.learn_insights = []
        self.verbose = verbose

    def fit(self, verbosity=0, validation_set=None):
        if self.verbose:
            print("Initializing U and V with random values")
        U = np.random.rand(self.m, self.rank)
        V = np.random.rand(self.n, self.rank)

        if self.method == "gd":
            E_old = np.zeros((self.m,self.n))
            for cycle in range(self.max_run + 1):
                R_predicted = np.matmul(U, V.T)
                E = (self.train_full - R_predicted)
                E[np.nonzero(self.train_full == 0)] = 0

                frobenius_error = 0.5* np.abs(np.sum(E*E))
                last_step = 0.5 * np.abs(np.sum(E*E) - np.sum(E_old**2))

                self.learn_insights.append((cycle, frobenius_error, last_step))

                if self.verbose and cycle % 10 == 0:
                    print(f"{cycle} cycles (of {self.max_run}) error (frobenius): {frobenius_error} last step size: {last_step}")

                if (last_step <= self.epsilon and self.convergence_check == "step") or (
                        frobenius_error <= self.epsilon and self.convergence_check == "value"):
                    if self.verbose:
                        print(f"Convergence reached after {cycle} cycles! Error (frobenius): {frobenius_error} last step size: {last_step}")
                    break

                U_old = U
                V_old = V
                U = U - self.eta * self.regularization * U_old + self.eta * np.matmul(E, V_old)
                V = V - self.eta * self.regularization * V_old + self.eta * np.matmul(E.T, U_old)
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

        self.U = U
        self.V = V
        self.matrix = np.matmul(U, V.T)

    def predict(self, coords):
        if not isinstance(coords, np.ndarray) or coords.shape[1] != 2:
            raise Exception("Predict parameter is not of type np.ndarray or has wrong column count")
        return self.matrix[coords[:,0],coords[:,1]]

class NMF:
    def __init__(self,          #TODO check each parameter for correct type
                    X_train,
                    rank=10,
                    random_state=random.randint(1, 101),
                    regularization=None,
                    epsilon=1,
                    stability = 10**(-9),
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
        self.stability = stability
        self.max_run = max_run
        self.regularization = regularization
        np.random.seed(self.random_state)
        self.matrix = None

    def fit(self, verbosity=0, validation_set = None):
        if verbosity == 2 and validation_set is None:
            raise Exception("For usage of verbosity level 2 an validation set is required")
        if verbosity >= 1:
            print("initialize U and V with random values")
        U = np.random.rand(self.m, self.rank)
        V = np.random.rand(self.n, self.rank)
        U_min = U
        V_min = V
        min_error = float("inf")
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
            U_delta_numerator = U_old * np.matmul(self.train_full,V)
            U_delta_denomerator = np.matmul(U_old, np.matmul(V.T, V)) + self.stability
            U = U_delta_numerator / U_delta_denomerator

            V_delta_numerator = V_old * np.matmul(self.train_full.T, U)
            V_delta_denomerator = np.matmul(V_old, np.matmul(U.T, U)) + self.stability
            V = V_delta_numerator/V_delta_denomerator
        self.matrix = np.matmul(U,V.T)

    def predict(self, coords):
        if not isinstance(coords, np.ndarray):
            raise Exception("Predict parameter is not of type np.ndarray or has wrong column count")
        return self.matrix[coords[:,0],coords[:,1]]