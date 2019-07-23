import numbers
import random

import numpy as np
import scipy

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
                    bias=False,
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
        if isinstance(eta, numbers.Number):
            self.eta = eta
        else:
            self.eta = -1

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
        self.bias = bias
        self.global_mean = 0

    def __loss_function(self, uv, user_count, k, ratings, lambd):
        U = uv[:user_count*k].reshape(-1, k)
        V = uv[user_count*k:].reshape(-1, k)
        predicted = np.matmul(U, V.T)
        E = ratings - predicted
        E[np.nonzero(ratings == 0)] = 0
        return 0.5 * np.sum(E * E) + 0.5*lambd * np.sum(U*U) + 0.5*lambd * np.sum(V*V)

    def __gradient_loss_function(self, uv, user_count, k, ratings, lambd):
        U = uv[:user_count * k].reshape(-1, k)
        V = uv[user_count * k:].reshape(-1, k)
        predicted = np.matmul(U, V.T)
        E = ratings - predicted
        E[np.nonzero(ratings == 0)] = 0
        gradU = lambd * U - np.matmul(E, V)
        gradV = lambd * V - np.matmul(E.T, U)
        return np.row_stack((gradU, gradV)).reshape(-1, 1).ravel()

    def fit(self, verbosity=0):
        if self.verbose:
            print("Initializing U and V with random values")
        if self.bias:
            self.rank = self.rank + 2
            U = np.random.rand(self.m, self.rank)
            U[:, self.rank - 1] = 1
            V = np.random.rand(self.n, self.rank)
            V[:, self.rank - 2] = 1
            global_sum = np.sum(self.train_full)
            entry_count = np.sum(self.train_full != 0)
            self.global_mean = global_sum/entry_count
            mean_free = self.train_full - self.global_mean
            mean_free[self.train_full == 0] = 0
            self.train_full = mean_free
        else:
            U = np.random.rand(self.m, self.rank)
            V = np.random.rand(self.n, self.rank)

        if self.method == "gd":
            E_old = np.zeros((self.m,self.n))
            U_old = U
            V_old = V
            for cycle in range(self.max_run + 1):
                R_predicted = np.matmul(U, V.T)
                E = (self.train_full - R_predicted)
                E[np.nonzero(self.train_full == 0)] = 0

                loss = self.__loss_function(np.row_stack((U, V)).reshape(-1,1).ravel(), self.m, self.rank, self.train_full, self.regularization)
                last_step = np.abs(self.__loss_function(np.row_stack((U, V)).reshape(-1,1).ravel(), self.m, self.rank, self.train_full, self.regularization) -
                                   self.__loss_function(np.row_stack((U_old, V_old)).reshape(-1,1).ravel(), self.m, self.rank, self.train_full, self.regularization))

                self.learn_insights.append((cycle, loss, last_step))

                #if self.verbose and cycle % 10 == 0:
                if self.verbose:
                    print(f"{cycle} cycles (of {self.max_run}) error (frobenius): {loss} last step size: {last_step}")

                if ((last_step <= self.epsilon and self.convergence_check == "step") or (
                        loss <= self.epsilon and self.convergence_check == "value")) and cycle != 0 :
                    if self.verbose:
                        print(f"Convergence reached after {cycle} cycles! Error (frobenius): {loss} last step size: {last_step}")
                    break

                U_old = U
                V_old = V
                UV = np.row_stack((U,V))

                if self.eta == -1:
                    #line_search
                    gradient = self.__gradient_loss_function(np.copy(UV).reshape(-1,1).ravel(), self.m,
                                                             self.rank, self.train_full, self.regularization)
                    UV = np.row_stack((U, V))
                    eta = scipy.optimize.line_search(self.__loss_function, self.__gradient_loss_function,
                                                     np.row_stack((U, V)).reshape(-1,1).ravel(), -gradient, c2=0.5,
                                                     args=(self.m, self.rank, self.train_full, self.regularization))[0]
                    if eta == None:
                        break
                else:
                    eta = self.eta


                UV = UV - eta * self.__gradient_loss_function(np.copy(UV).reshape(-1,1).ravel(), self.m, self.rank, self.train_full,
                                                                self.regularization).reshape(-1, self.rank)
                U = UV[:self.m,:]
                V = UV[self.m:,:]
                if self.bias:
                    U[:, self.rank - 1] = 1
                    V[:, self.rank - 2] = 1
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
        if self.bias:
            return self.matrix[coords[:,0],coords[:,1]] + self.global_mean
        else:
            return self.matrix[coords[:,0],coords[:,1]]

class NMF:
    def __init__(self,          #TODO check each parameter for correct type
                    X_train,
                    rank=10,
                    epsilon=1,
                    stability = 10**(-9),
                    max_run=1_000,
                    verbose=False
                ):
        if X_train.shape[1] != 3:
            raise Exception("wrong shape")
        self.train = X_train
        self.train_full = convert_sparse_coo_to_full_matrix(X_train).toarray()
        self.m = self.train_full.shape[0]
        self.n = self.train_full.shape[1]
        self.rank = rank
        self.epsilon = epsilon
        self.stability = stability
        self.max_run = max_run
        self.matrix = None
        self.verbose = verbose

    def fit(self, verbosity=0, validation_set = None):
        if verbosity == 2 and validation_set is None:
            raise Exception("For usage of verbosity level 2 an validation set is required")
        if self.verbose:
            print("initialize U and V with random values")
        U = np.random.rand(self.m, self.rank)
        V = np.random.rand(self.n, self.rank)
        E_old = np.matmul(U, V.T)
        for i in range(0, self.max_run+1):
            R_predicted = np.matmul(U, V.T)
            E = self.train_full - R_predicted
            E[np.nonzero(self.train_full == 0)] = 0
            error = 0.5* np.sum(E*E)
            last_step = 0.5 * np.abs(np.sum(E * E) - np.sum(E_old ** 2))
            if self.verbose and i % 10 == 0:
                print(f"Cycle {i} of {self.max_run}; error: {error}, step_size: {last_step}")
            if last_step <= self.epsilon:
                break
            U_old = U
            V_old = V
            U_delta_numerator = U_old * np.matmul(self.train_full,V)
            U_delta_denomerator = np.matmul(U_old, np.matmul(V.T, V)) + self.stability
            U = U_delta_numerator / U_delta_denomerator

            V_delta_numerator = V_old * np.matmul(self.train_full.T, U)
            V_delta_denomerator = np.matmul(V_old, np.matmul(U.T, U)) + self.stability
            V = V_delta_numerator/V_delta_denomerator
            E_old = E
        self.U = U
        self.V = V
        self.matrix = np.matmul(U,V.T)

    def predict(self, coords):
        if not isinstance(coords, np.ndarray):
            raise Exception("Predict parameter is not of type np.ndarray or has wrong column count")
        return self.matrix[coords[:,0],coords[:,1]]